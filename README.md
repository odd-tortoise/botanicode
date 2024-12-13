# Tomato JSON Configuration

This JSON file contains configuration data for simulating the growth of a tomato plant. The data is organized into several sections, each specifying different parameters related to the plant's growth and initial conditions.

## JSON Structure

The JSON file is structured as follows:

```json
{
  "phylotaxis": {
    "leaflets_number": <integer>,
    "leaf_arrangement": <string>,
    "angle": <float>
  },
  "growth_data": {
    "internode_length_max": <float>,
    "internode_radius_max": <float>,
    "k_internodes": <float>,
    "leaves_size_max": <float>,
    "k_leaves": <float>,
    "internode_appereace_rate": <float>
  },
  "initial_data": {
    "initial_stem_length": <float>,
    "initial_stem_radius": <float>,
    "initial_leaf_number": <integer>,
    "initial_leaflets_number": <integer>,
    "initial_root_length": <float>,
    "initial_root_radius": <float>
  }
}
```

## Fields Explanation

### Phylotaxis

- **leaflets_number**: The number of leaflets per leaf.
- **leaf_arrangement**: The arrangement of leaves on the stem. Possible values include "alternate", "opposite", and "decussate".
- **angle**: The angle between leaves in the alternate arrangement, specified in degrees, MUST be specified for the "alternate" pattern

### Growth Data

- **internode_length_max**: The maximum length of the internodes (the segments between nodes on the stem).
- **internode_radius_max**: The maximum radius of the internodes.
- **k_internodes**: The growth rate constant for internodes.
- **leaves_size_max**: The maximum size of the leaves.
- **k_leaves**: The growth rate constant for leaves.
- **internode_appereace_rate**: The rate at which new internodes appear, specified in time units.

### Initial Data

- **initial_stem_length**: The initial length of the stem.
- **initial_stem_radius**: The initial radius of the stem.
- **initial_leaf_number**: The initial number of leaves.
- **initial_leaflets_number**: The initial number of leaflets per leaf.
- **initial_root_length**: The initial length of the root.
- **initial_root_radius**: The initial radius of the root.

## Example

Here is an example of a 

tomato.json

 file:

```json
{
  "phylotaxis": {
    "leaflets_number": 5,
    "leaf_arrangement": "alternate",
    "angle": 137.5
  },
  "growth_data": {
    "internode_length_max": 5,
    "internode_radius_max": 1,
    "k_internodes": 0.2,
    "leaves_size_max": 10,
    "k_leaves": 0.1,
    "internode_appereace_rate": 3
  },
  "initial_data": {
    "initial_stem_length": 3,
    "initial_stem_radius": 0.1,
    "initial_leaf_number": 2,
    "initial_leaflets_number": 1,
    "initial_root_length": 1,
    "initial_root_radius": 0.1
  }
}
```

This example specifies the initial conditions and growth parameters for simulating the growth of a tomato plant. Adjust the values as needed to fit your specific simulation requirements.