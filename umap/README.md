# README: Parameter Usage for `config.json`

- **aggregate_on_collection**: Boolean flag to aggregate embeddings at the collection level (`true`) or conversation level (`false`).

- **show_only**: Filters the data to show only a specific group. Possible values: `"facilitators"`, `"participants"`, or `"all"`.

- **aggregate_embeddings**: (`true`) will aggreagte embeddings according to **aggregate_on_collection**. Set to (`false`) overrides aggregate_on_collection and embeds each individual SpeakerTurn.

- **truncate_turns**: Boolean flag to truncate speaker turns to the middle quartiles (`true`) or show all turns (`false`). Amount can be adjusted inline code.

- **umap_params**: Parameters for UMAP dimensionality reduction. Includes:
  - **n_components**: Number of UMAP components (default: `3` for 3D plots).
  - **random_state**: Random seed for reproducibility.
  - **n_neighbors**: Number of neighbors for UMAP.
  - **metric**: Distance metric for UMAP.

- **supervised_umap**: Settings for supervised UMAP:
  - **enabled**: Whether to use supervised UMAP (`true`/`false`).
  - **label_column**: The column used for labels in supervised UMAP. Example: `"collection_id"` or `"conversation_id"`.

- **color_by_role**: Specifies how to color the plot, either by role (`"f_p"`) or collection (`"collection"`).