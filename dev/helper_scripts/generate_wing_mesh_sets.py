"""Generate the standard uniform and clustered RA10 wing mesh families."""

from pathlib import Path

from wing_creator import create_wing, export_wing_to_stl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MESH_ROOT = PROJECT_ROOT / "meshes"

# With open tips, these station pairs give 192, 672, 2,208, 7,360, and
# 24,000 surface facets.
RESOLUTIONS = ((4, 9), (7, 15), (13, 24), (24, 41), (41, 76))


def generate_mesh_sets():
    outputs = []
    for set_name, clustered in (("uniform", False), ("clustered", True)):
        output_directory = MESH_ROOT / "RA10_RT0.5_span10_sweep0" / set_name
        for n_span, n_chord in RESOLUTIONS:
            wing = create_wing(
                RA=10.0,
                RT=0.5,
                b=10.0,
                series="0010",
                n_span=n_span,
                n_chord=n_chord,
                span_cluster=clustered,
                chord_cluster=clustered,
                sweep=0.0,
                close_tips=False,
            )
            output = output_directory / f"wing_{len(wing):05d}_panels.stl"
            export_wing_to_stl(wing, output)
            outputs.append(output)
    return outputs


if __name__ == "__main__":
    for generated_file in generate_mesh_sets():
        print(generated_file.relative_to(PROJECT_ROOT))
