"""Convert the matplotlib Figure 1 (fig_overview.py) into an editable draw.io file.
Text, boxes, rectangles, circles, arrows and short lines become native draw.io cells; long polylines (waveforms,
curves) and polygons become small embedded SVG images.
usage: python fig_to_drawio.py <fig_overview.py> <out.drawio>
"""
import sys, os, runpy, base64, tempfile, re
from xml.sax.saxutils import escape
import numpy as np
import matplotlib.colors as mc
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch

src, out = sys.argv[1], sys.argv[2]
tmp = tempfile.mkdtemp()
sys.argv = [src, tmp]
g = runpy.run_path(src)
ax, W, H = g["ax"], g["W"], g["H"]

S = 10.0                      # px per data unit (canvas 1400 x 620 px)
PT = 200.0 / 72.0             # figure is 7 in = 1400 px, so 1 pt = 2.78 px
X = lambda u: u * S
Y = lambda v: (H - v) * S

cells = []; nid = [1]
def new_id():
    nid[0] += 1; return f"c{nid[0]}"


def hexc(c):
    return mc.to_hex(c, keep_alpha=False).upper() if c is not None and mc.to_rgba(c)[3] > 0 else "none"


def alpha_of(c, a=None):
    # matplotlib already folds the artist alpha into get_facecolor(); use it once
    if a is not None:
        return int(round(100 * a))
    return int(round(100 * (mc.to_rgba(c)[3] if c is not None else 1.0)))


def vertex(x, y, w, h, style, value=""):
    cells.append(f'<mxCell id="{new_id()}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')


def edge(pts, style):
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    mids = "".join(f'<mxPoint x="{px:.1f}" y="{py:.1f}"/>' for px, py in pts[1:-1])
    arr = f'<Array as="points">{mids}</Array>' if mids else ""
    cells.append(f'<mxCell id="{new_id()}" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{x0:.1f}" y="{y0:.1f}" as="sourcePoint"/><mxPoint x="{x1:.1f}" y="{y1:.1f}" as="targetPoint"/>'
                 f'{arr}</mxGeometry></mxCell>')


def svg_image(svg_body, x0, y0, x1, y1):
    w, h = x1 - x0, y1 - y0
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.1f}" height="{h:.1f}" viewBox="{x0:.1f} {y0:.1f} {w:.1f} {h:.1f}">'
           f'{svg_body}</svg>')
    b64 = base64.b64encode(svg.encode()).decode()
    vertex(x0, y0, w, h, f"shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;html=1;"
                         f"image=data:image/svg+xml,{b64};")


def dash(ls, lw):
    if isinstance(ls, tuple) or ls in ("--", "dashed", ":", "dotted", "-."):
        return "dashed=1;dashPattern=3 2;"
    return ""


# ---- patches (z-order as drawn)
items = sorted(list(ax.patches) + list(ax.lines) + list(ax.texts), key=lambda a: a.get_zorder())
for a in items:
    if isinstance(a, FancyArrowPatch):
        (x0, y0), (x1, y1) = a._posA_posB
        c = hexc(a.get_edgecolor()); lw = a.get_linewidth() * PT
        cs = a.get_connectionstyle()
        rad = getattr(cs, "rad", 0.0)
        pts = [(X(x0), Y(y0)), (X(x1), Y(y1))]
        style = (f"endArrow=block;endFill=1;endSize=5;html=1;rounded=1;strokeColor={c};strokeWidth={lw:.1f};")
        if abs(rad) > 1e-6:   # arc3: control point at the midpoint, offset by rad * length perpendicular
            mx_, my_ = (x0 + x1) / 2, (y0 + y1) / 2; dx, dy = x1 - x0, y1 - y0
            cx, cy = mx_ + 0.5 * rad * dy, my_ - 0.5 * rad * dx   # curve apex of the quadratic Bezier
            pts = [pts[0], (X(cx), Y(cy)), pts[1]]
            style += "curved=1;"
        edge(pts, style)
    elif isinstance(a, FancyBboxPatch):
        x, y, w, h = a.get_x(), a.get_y(), a.get_width(), a.get_height()
        fc, ec = a.get_facecolor(), a.get_edgecolor()
        r = a.get_boxstyle().rounding_size or 0
        arc = max(1, min(50, int(100 * r / min(w, h))))
        vertex(X(x), Y(y + h), w * S, h * S,
               f"rounded=1;arcSize={arc};whiteSpace=wrap;html=1;fillColor={hexc(fc)};strokeColor={hexc(ec)};"
               f"strokeWidth={a.get_linewidth() * PT:.1f};opacity={alpha_of(fc, a.get_alpha())};"
               + ("dashed=1;" if a.get_linestyle() not in ("-", "solid") else ""))
    elif isinstance(a, Rectangle):
        x, y, w, h = a.get_x(), a.get_y(), a.get_width(), a.get_height()
        fc, ec = a.get_facecolor(), a.get_edgecolor()
        lw = a.get_linewidth() * PT
        vertex(X(x), Y(y + h), w * S, h * S,
               f"rounded=0;html=1;fillColor={hexc(fc)};strokeColor={hexc(ec) if lw > 0 else 'none'};"
               f"strokeWidth={lw:.1f};fillOpacity={alpha_of(fc, a.get_alpha())};")
    elif isinstance(a, Circle):
        (cx, cy), r = a.center, a.radius
        vertex(X(cx - r), Y(cy + r), 2 * r * S, 2 * r * S,
               f"ellipse;html=1;fillColor={hexc(a.get_facecolor())};strokeColor={hexc(a.get_edgecolor())};"
               f"strokeWidth={a.get_linewidth() * PT:.1f};opacity={alpha_of(a.get_edgecolor(), a.get_alpha())};")
    elif isinstance(a, Polygon):
        xy = a.get_xy()
        pts = " ".join(f"{X(px):.1f},{Y(py):.1f}" for px, py in xy)
        lw = a.get_linewidth() * PT
        xs_, ys_ = [X(p[0]) for p in xy], [Y(p[1]) for p in xy]
        svg_image(f'<polygon points="{pts}" fill="{hexc(a.get_facecolor())}" stroke="{hexc(a.get_edgecolor())}" '
                  f'stroke-width="{lw:.2f}" stroke-linejoin="round"/>',
                  min(xs_) - lw, min(ys_) - lw, max(xs_) + lw, max(ys_) + lw)
    elif a.__class__.__name__ == "Line2D":
        xd, yd = np.asarray(a.get_xdata(), float), np.asarray(a.get_ydata(), float)
        c = hexc(a.get_color()); lw = a.get_linewidth() * PT; op = alpha_of(a.get_color(), a.get_alpha())
        if len(xd) <= 3:
            edge([(X(x), Y(y)) for x, y in zip(xd, yd)],
                 f"endArrow=none;html=1;rounded=0;strokeColor={c};strokeWidth={lw:.1f};opacity={op};" + dash(a.get_linestyle(), lw))
        else:
            px, py = X(xd), Y(yd)
            pts = " ".join(f"{u:.2f},{v:.2f}" for u, v in zip(px, py))
            da = ' stroke-dasharray="3,2"' if dash(a.get_linestyle(), lw) else ""
            svg_image(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-opacity="{op / 100:.2f}" '
                      f'stroke-width="{lw:.2f}" stroke-linejoin="round" stroke-linecap="round"{da}/>',
                      px.min() - lw, py.min() - lw, px.max() + lw, py.max() + lw)
    elif a.__class__.__name__ == "Text":
        s = a.get_text()
        if not s.strip():
            continue
        # mathtext $...$ -> MathJax inline \( ... \)
        s = re.sub(r"\$(.+?)\$", lambda m: r"\(" + m.group(1).replace(r"\!", "") + r"\)", s)
        s = escape(s).replace("\n", "<br>")
        fs = a.get_fontsize() * PT
        lines = a.get_text().split("\n")
        w = max(len(l) for l in lines) * fs * 0.62 + 12
        h = len(lines) * fs * 1.25 + 4
        x, y = a.get_position()
        ha, va = a.get_ha(), a.get_va()
        left = X(x) - (w / 2 if ha == "center" else (w if ha == "right" else 0))
        top = Y(y) - (h / 2 if va == "center" else 0)
        fstyle = (1 if a.get_fontweight() in ("bold", 700) else 0) + (2 if a.get_style() == "italic" else 0)
        cells.append(f'<mxCell id="{new_id()}" value="{escape(s, {chr(34): "&quot;"})}" '
                     f'style="text;html=1;whiteSpace=nowrap;align={ha if ha != "center" else "center"};verticalAlign=middle;'
                     f'fontFamily=Helvetica;fontSize={fs:.0f};fontColor={hexc(a.get_color())};fontStyle={fstyle};'
                     f'spacing=0;" vertex="1" parent="1">'
                     f'<mxGeometry x="{left:.1f}" y="{top:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')

xml = (f'<mxfile host="app.diagrams.net"><diagram name="Figure 1" id="fig1">'
       f'<mxGraphModel dx="{W * S:.0f}" dy="{H * S:.0f}" grid="1" gridSize="5" guides="1" page="1" pageScale="1" '
       f'pageWidth="{W * S:.0f}" pageHeight="{H * S:.0f}" math="1" background="#FFFFFF"><root>'
       f'<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')
open(out, "w").write(xml)
print("cells", len(cells), "->", out, os.path.getsize(out) // 1024, "KB")
