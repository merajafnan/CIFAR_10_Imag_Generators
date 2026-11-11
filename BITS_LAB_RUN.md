# Final run on BITS Prayogshala

Copy this folder to the assigned BITS Prayogshala GPU machine. The assignment requires this infrastructure. The full run will take several hours. Do not set `UDL_QUICK=1` for the submission run.

The supplied HTML files and `PartC_Comparative_Analysis_PREVIEW.pdf` show the saved run from the original notebook. Replace the HTML files after the lab run. Submit the new `PartC_Comparative_Analysis.pdf`, not the preview.

## 1. Record lab and GPU evidence

Open a terminal in this folder. Run `hostname`, `pwd`, and `nvidia-smi`. Keep the BITS lab page or session name visible. Save a screenshot as `BITS_Prayogshala_GPU.jpg`. The screenshot must show that the assigned lab GPU is in use. Do not include passwords or access tokens.

## 2. Create the environment

Use the Python version supplied by the lab. From this folder, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last command must print `True` for CUDA. If the lab has a preinstalled CUDA build of PyTorch, use that build. Consult the lab's setup instructions if `torch.cuda.is_available()` is `False`.

## 3. Run both notebooks in order

Use Jupyter with this environment. Open `UDL_Assignment2_PartA.ipynb`. Select its Python environment. Choose **Kernel > Restart Kernel and Run All Cells**. Wait for every cell to finish, then save the notebook. Repeat for `UDL_Assignment2_PartB.ipynb`.

The notebooks write metrics and 100-image grids to `results/`. The full run uses 10,000 generated images for final FID. Check that these four files exist:

```bash
ls results/figs/vae_beta1_100.png \
   results/figs/vqvae_k256_100.png \
   results/figs/wgan_100.png \
   results/figs/wgan-gp_100.png
```

## 4. Export the HTML files and Part C PDF

Run:

```bash
python update_observations.py
python -m nbconvert --to html UDL_Assignment2_PartA.ipynb
python -m nbconvert --to html UDL_Assignment2_PartB.ipynb
python build_part_c.py
```

The first command updates the notebook observations from the lab results. Open both HTML files and `PartC_Comparative_Analysis.pdf`. Check the group details, results, and all 100-image grids. The metrics in the PDF must match the lab notebook outputs.

## 5. Submit one group package

Submit these files for Group 9:

- `UDL_Assignment2_PartA.ipynb`
- `UDL_Assignment2_PartA.html`
- `UDL_Assignment2_PartB.ipynb`
- `UDL_Assignment2_PartB.html`
- `PartC_Comparative_Analysis.pdf`
- `BITS_Prayogshala_GPU.jpg`

Keep the `results/` folder and `requirements.txt` with your working copy. They support the reported figures and a repeat run.
