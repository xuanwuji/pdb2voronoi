# pdb2voronoi
This is a module that can convert protein PDB format files into Voronoi diagrams

### Input
- Required parameter: Path to the PDB file
- Optional parameters: Path to the optional residue file (residue_list.csv), as well as the optional PDB ID in the residue

#### Note:
The optional residue file (residue_list.csv) should be in CSV format and must include at least two attributes: pdbid and residueid.

### Output
- **figure1**: Represents the projection onto a plane, which is the plane fitted using the least squares method.
- **figure2**: Represents the projection onto a Miller cylindrical surface.
- **figure3**: Displays both figures in a single window.
- The generated Voronoi diagram: Red denotes acidic amino acids, blue denotes basic amino acids, and green denotes other amino acids.

### Usage
1. Import the module:
    ```python
    import pdbToVoronoi
    ```

2. Call the function:
    ```python
    pdbToVoronoi.pdbToVoronoi("xxx.pdb", "residue_list.csv"=None, "pdbid"=None)
    ```

3. Generate the images:
    - `figure1`
    - `figure2`
    - `figure3`
