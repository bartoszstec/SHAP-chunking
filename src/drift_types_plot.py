import matplotlib.pyplot as plt
import numpy as np

def format_axes(ax):
    """Wspólne formatowanie osi jako strzałek."""
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)

    # Stylizacja osi jako strzałek
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Ukrycie domyślnych podziałek (ticks)
    ax.set_xticks([])
    ax.set_yticks([])

    # Dodanie strzałek na końcach osi
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    # Etykiety osi
    ax.set_xlabel('Czas', loc='right', labelpad=15, fontsize=10, fontweight='bold')
    ax.set_ylabel('Koncept', loc='top', labelpad=15, fontsize=10, fontweight='bold')


# ----------------------------------------------------
# (a) Dryf nagły
# ----------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(5, 4))

ax1.scatter(range(1, 7), [1] * 6, color='red', marker='o', s=50, zorder=3)
ax1.scatter(range(7, 13), [3] * 6, color='#3399FF', marker='s', s=50, zorder=3)

ax1.set_title('(a) Dryf nagły', y=-0.25, fontsize=11, fontweight='bold')
format_axes(ax1)

plt.tight_layout()
# plt.savefig('abrupt_drift.pdf', bbox_inches='tight')
plt.savefig('abrupt_drift.png', dpi=300, bbox_inches='tight')
plt.show()


# ----------------------------------------------------
# (b) Dryf stopniowy
# ----------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(5, 4))

x_red_b = [1, 2, 3, 6, 8, 9]
y_red_b = [1] * len(x_red_b)
x_blue_b = [4, 5, 7, 10, 11, 12]
y_blue_b = [3] * len(x_blue_b)

ax2.scatter(x_red_b, y_red_b, color='red', marker='o', s=50, zorder=3)
ax2.scatter(x_blue_b, y_blue_b, color='#3399FF', marker='s', s=50, zorder=3)

ax2.set_title('(b) Dryf stopniowy', y=-0.25, fontsize=11, fontweight='bold')
format_axes(ax2)

plt.tight_layout()
# plt.savefig('gradual_drift.pdf', bbox_inches='tight')
plt.savefig('gradual_drift.png', dpi=300, bbox_inches='tight')
plt.show()


# ----------------------------------------------------
# (c) Dryf inkrementalny
# ----------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(5, 4))

ax3.scatter(range(1, 5), [1] * 4, color='red', marker='o', s=50, zorder=3)

x_trans = [5, 6, 7, 8]
y_trans = [1.5, 1.9, 2.3, 2.7]
colors_trans = ['#E65100', '#F57C00', '#FFB74D', '#BA68C8']
markers_trans = ['o', 'o', 's', 's']

for x, y, c, m in zip(x_trans, y_trans, colors_trans, markers_trans):
    ax3.scatter(x, y, color=c, marker=m, s=50, zorder=3)

ax3.scatter(range(9, 13), [3] * 4, color='#3399FF', marker='s', s=50, zorder=3)

ax3.set_title('(c) Dryf inkrementalny', y=-0.25, fontsize=11, fontweight='bold')
format_axes(ax3)

plt.tight_layout()
#plt.savefig('incremental_drift.pdf', bbox_inches='tight')
plt.savefig('incremental_drift.png', dpi=300, bbox_inches='tight')
plt.show()