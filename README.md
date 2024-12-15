# Tomato JSON Configuration

This JSON file contains configuration data for simulating the growth of a tomato plant. The data is organized into several sections, each specifying different parameters related to the plant's growth and initial conditions.

## JSON Structure

The JSON file is structured as follows:

```json
{
  "general": {
    "common_name": <string>,
    "scientific_name": <string>
  },
  "phylotaxis": {
    "leaflets_number": <integer>,
    "leaf_arrangement": <string>,
    "angle": <float>
  },
  "leaves_paramters": {
    "leaves_number": <integer>,
    "new_leaf_size": <float>,
    "new_petioles_size": <float>,
    "leaf_y_angle": <float>
  },
  "stem_parameters": {
    "new_stem_lenght": <float>,
    "new_stem_radius": <float>
  },
  "root_parameters": {
    "new_root_lenght": <float>,
    "new_root_radius": <float>
  },
  "growth_data": {
    "internode_lenght_max": <float>,
    "internode_radius_max": <float>,
    "k_internodes": <float>,
    "leaves_size_max": <float>,
    "k_leaves": <float>,
    "internode_appereace_rate": <float>,
    "plant_height_max": <float>
  },
  "initial_data": {
    "initial_stem_lenght": <float>,
    "initial_stem_radius": <float>,
    "initial_leaf_number": <integer>,
    "initial_leaflets_number": <integer>,
    "initial_root_lenght": <float>,
    "initial_root_radius": <float>
  }
}
```

## Fields Explanation

### General

- **common_name**: The common name of the plant.
- **scientific_name**: The scientific name of the plant.

### Phylotaxis

- **leaflets_number**: The number of leaflets per leaf.
- **leaf_arrangement**: The arrangement of leaves on the stem. Possible values include "alternate", "opposite", and "decussate".
- **angle**: The angle between leaves in the alternate arrangement, specified in degrees, MUST be specified for the "alternate" pattern

### Leaves Parameters

- **leaves_number**: The number of leaves.
- **new_leaf_size**: The size of new leaves.
- **new_petioles_size**: The size of new petioles.
- **leaf_y_angle**: The angle of the leaves in the Y direction.

### Stem Parameters

- **new_stem_lenght**: The length of new stems.
- **new_stem_radius**: The radius of new stems.

### Root Parameters

- **new_root_lenght**: The length of new roots.
- **new_root_radius**: The radius of new roots.

### Growth Data

- **internode_lenght_max**: The maximum length of the internodes (the segments between nodes on the stem).
- **internode_radius_max**: The maximum radius of the internodes.
- **k_internodes**: The growth rate constant for internodes.
- **leaves_size_max**: The maximum size of the leaves.
- **k_leaves**: The growth rate constant for leaves.
- **internode_appereace_rate**: The rate at which new internodes appear, specified in time units.
- **plant_height_max**: The maximum height of the plant.

### Initial Data

- **initial_stem_lenght**: The initial length of the stem.
- **initial_stem_radius**: The initial radius of the stem.
- **initial_leaf_number**: The initial number of leaves.
- **initial_leaflets_number**: The initial number of leaflets per leaf.
- **initial_root_lenght**: The initial length of the root.
- **initial_root_radius**: The initial radius of the root.

## Example

Here is an example of a 

tomato.json

 file:

```json
{
  "general": {
    "common_name": "Tomato",
    "scientific_name": "Solanum lycopersicum"
  },
  "phylotaxis": {
    "leaflets_number": 5,
    "leaf_arrangement": "alternate",
    "angle": 137.5
  },
  "leaves_paramters": {
    "leaves_number": 10,
    "new_leaf_size": 5,
    "new_petioles_size": 2,
    "leaf_y_angle": 45
  },
  "stem_parameters": {
    "new_stem_lenght": 3,
    "new_stem_radius": 0.5
  },
  "root_parameters": {
    "new_root_lenght": 4,
    "new_root_radius": 0.3
  },
  "growth_data": {
    "internode_lenght_max": 5,
    "internode_radius_max": 1,
    "k_internodes": 0.2,
    "leaves_size_max": 10,
    "k_leaves": 0.1,
    "internode_appereace_rate": 3,
    "plant_height_max": 100
  },
  "initial_data": {
    "initial_stem_lenght": 3,
    "initial_stem_radius": 0.1,
    "initial_leaf_number": 2,
    "initial_leaflets_number": 1,
    "initial_root_lenght": 1,
    "initial_root_radius": 0.1
  }
}
```

This example specifies the initial conditions and growth parameters for simulating the growth of a tomato plant. Adjust the values as needed to fit your specific simulation requirements.