# ai-practicum

## Generating the Proposal PDF

After installing `latexmk` from [here](https://mg.readthedocs.io/latexmk.html), you should be able to generate the PDF from the latex file.

Simply clone the repo and then run the following commands:

```bash
cd proposal
make
```

This will run `latexmk` with the proper configurations using the `Makefile`. Running `latexmk` in this way puts it in **_hot-compilation_** mode which means that whenever you save the file, it will recompile the latex into a PDF. The recommended workflow is to download the LaTex Workshop extension on VS Code (extension ID: `James-Yu.latex-workshop`). Once installed, a **_TEX_** button will show up on the left side of your VS Code which you can use the view the PDF in real-time.

## Data
### Initial Data Links:
Metadata: https://downloads.loni.usc.edu/download/image-metadata?key=bf93383e-4366-4ae4-819e-5ff82abbd748&zip=Initial%20AD/CN%20fmri%20set_metadata.zip
Data: https://downloads.loni.usc.edu/download/files/ida1/bf93383e-4366-4ae4-819e-5ff82abbd748/Initial%20AD/CN%20fmri%20set.zip
