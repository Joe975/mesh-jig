# Sky whale / manta

A friendly cel-shaded blue sky whale with one pair of broad manta pectoral fins, a whale muzzle and a horizontal two-lobed fluke. Preserve the silhouette and broad color zones of the frozen sheet.

- Rounded whale head and full ivory lower jaw/belly; blue upper body tapers continuously into a narrow tail peduncle. Nose is Blender +Y, up +Z, anatomical left X<0. Center the overall bounds at the origin.
- Exactly two small dark eyes beside the muzzle, with tiny pale highlights. A thin mouth boundary and understated ivory throat grooves support the whale face.
- One broad manta fin on each side. The top view defines swept planform and taper. The front defines the downward curved thickness profile and tip elevation. Fins have real thickness and smooth enough root transitions to deform.
- One horizontal whale fluke with two lobes and a central notch; thin in side/rear views, broad from above. No extra dorsal fins or appendages.
- Blue dorsum, deep blue shaded accents, ivory underside, cyan flowing stripes on back/fins/tail. Use spatially coherent markings, never reference-pixel projection or camera-specific paint. Do not model the illustration's light gradients as arbitrary patches.

## Frozen reconciliation and scale

Use one master physical scale: the left profile's 463 px nose-to-tail extent is 2.0 m (4.319654 mm/px). Front span 478 px gives 2.064795 m; front thickness 156 px gives 0.673866 m. Do not fit physical scale independently per view. Top depth 459 px and span 464 px are illustrative discrepancies, not a second size contract. Bounds tolerance is 5% on all three dimensions.

Front elevation and top planform take precedence for a single coherent symmetric 3D fin curve. Side fin tips remain approximately 10–15 px lower in the drawing (43–65 mm at this scale); do not bend the fins differently for side cameras or alter the reference. Cyan line layout is approximate across views. Hero is a visual design check, excluded from numeric score.

The asset budget is 100–16,000 triangles and one 1024 px texture the script paints, with mesh nodes preserved for rigging. Closed finite-thickness surfaces and economical topology are preferred. Preserve enough fin-root and tail geometry for blended weights. Construction method is the builder's choice.

## Comparison protocol

Use five successfully evaluated complete modeling attempts, fresh scripts for this run, immutable attempt directories, and preserve all failures. Do not read sibling builders' work. Run local mesh-jig commands only: no API agent loop, paid judge, reviewer or credentials. Inspect all comparisons, record honest visual critique, and separate visual selection, numeric best and latest. Freeze the selected source GLB by SHA256 before rigging; never rebuild it during the rig phase.

All five orthographic views have equal scoring weight. Numeric score is the equal mean of overlap, color and edges; resemblance >=0.5 is a separate diagnostic gate. Assess visible anatomy, silhouette, face, palette and markings across all views. Numbers do not certify appearance or animation quality.
