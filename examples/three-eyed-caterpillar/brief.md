# Three-eyed fronded caterpillar: frozen sweep contract

Build the creature in sheet.png. Blender metres, +Z up, +Y toward its face,
anatomical left X<0. Centre the total bounding box on the origin. Ground is
approximately Z=-0.24454343. No scene, floor, labels, backdrop or external image.

- Low indigo/teal segmented body tapering behind a round mask-like head. Use a
  head, six trunk lobes and one blunt terminal tail lobe. Seams are shallow rounded
  constrictions, not disconnected floating balls. Six bilateral pairs of short
  dark prolegs sit along the belly; the head and terminal tail have none.
- FRONT governs the triangular three-eye face: one centred upper eye and two
  lower lateral eyes, dark convex pupils, cyan rims and small pale highlights.
  Keep all three distinct, the pale lilac central facial mask and tiny dark mouth.
  No eyes or mouth on the back; TOP must not show the frontal eyes through skin.
- Exactly four bilateral feather-frond pairs. TOP governs their fore/aft stations;
  rounded station values from the sheet are Y=+0.25,+0.10,-0.05,-0.20 m.
  Roots lie on both shoulders of the dorsal body, approximately X=+/-0.10 m.
  Fronds splay outward, curve toward the tail, and have a slender shaft with many
  pointed lilac/cyan vanes and sparse pale sparkle accents. They must read as
  feather fronds, not cylindrical antlers or plain solid paddles.
- RIGHT governs length, belly pose and all frond heights. Use one scale:
  0.9/449 m per original sheet pixel. Its total height 244 px gives 0.48908686 m;
  TOP width 336 px gives 0.67349666 m. Length is 0.9 m. No per-view rescaling.
  FRONT and LEFT/REAR remain visibly inconsistent in height; do not silently
  average them or raise all fronds to match those views.
- Front-to-back frond tip heights above ground are approximately
  0.489,0.481,0.469,0.429 m (RIGHT peaks near sheet rows 586,590,596,616).
  Root heights above ground are approximately 0.22,0.22,0.205,0.19 m.
  TOP sets root spacing when side drawings disagree. These approximate landmarks
  fix anatomy and proportion; detailed shapes remain the builder's work.
- Body maximum height above ground is about 0.26 m; head width about 0.38 m.
  Behind the head the trunk tapers toward a tail about half the head width.
  Indigo shadow, violet transitions, teal/cyan dorsal scales; fronds brighter
  lilac/cyan, mask pale lilac, pupils near black. Use the frozen palette in jig.json.

Budget: 1,000–32,000 triangles, at least five vertex colours. Keep semantic node
names so the later rig can identify head/eyes, trunk, legs and eight fronds.
Separate meshes or joined skin are allowed. No fixed construction recipe.
All five orthographic views count equally; HERO is visual evidence only.
Numeric score is (2*overlap + zones + edges)/4. Regional views are observations,
not pass gates. The resemblance >=0.4 gate is diagnostic, not a delivery promise.
The generic orthographic outline diagnostic is used: no upright standing-table
calibration is asserted for this inconsistent, low horizontal creature.

Visual review priorities, in order: recognizable face and correct eye count;
eight paired fronds with convincing feather structure and attachment; low tapered
body silhouette and short prolegs; indigo/teal and lilac/cyan colour placement;
clean surfaces, no floating pieces and a riggable body. Numeric score is not a
visual ranking.
