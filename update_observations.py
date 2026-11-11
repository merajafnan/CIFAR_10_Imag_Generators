"""Refresh notebook discussion from the completed full-run result JSON files."""

import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def replace_markdown(notebook, heading, body):
    matches = [cell for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith(heading)]
    if len(matches) != 1:
        raise SystemExit(f"Expected one {heading!r} section")
    matches[0].source = body


def main():
    vae = json.loads((RESULTS / "partA_vae.json").read_text())
    vq = json.loads((RESULTS / "partA_vqvae.json").read_text())
    gan = json.loads((RESULTS / "partB_wgan.json").read_text())
    if any(len(vae[b]["psnr_curve"]) != 40 for b in ("1", "2", "4", "10")):
        raise SystemExit("Run the full Part A notebook before updating observations")
    if any(len(vq[k]["psnr_curve"]) != 30 or len(vq[k]["pcnn_curve"]) != 20 for k in ("512", "256", "128")):
        raise SystemExit("Run the full Part A notebook before updating observations")
    if any(gan[k]["fid_it"][-1] != 40000 for k in ("wgan", "wgan-gp")):
        raise SystemExit("Run the full Part B notebook before updating observations")

    part_a_path = ROOT / "UDL_Assignment2_PartA.ipynb"
    part_b_path = ROOT / "UDL_Assignment2_PartB.ipynb"
    part_a = nbformat.read(part_a_path, as_version=4)
    part_b = nbformat.read(part_b_path, as_version=4)
    b1, b10 = vae["1"], vae["10"]
    replace_markdown(part_a, "### β-VAE observations", f"""### β-VAE observations

At β=1, test PSNR was {b1['final_psnr']:.2f} dB and sample FID was {b1['fid_samples']:.2f}. At β=10, test PSNR was {b10['final_psnr']:.2f} dB and sample FID was {b10['fid_samples']:.2f}. The KL term was {b1['kl']:.2f} per image at β=1 and {b10['kl']:.2f} at β=10. Compare the PSNR curve, FID table, reconstructions, and sample grids to judge how β affects detail and sample quality.
""")
    k512, k256, k128 = (vq[k] for k in ("512", "256", "128"))
    replace_markdown(part_a, "### VQ-VAE observations", f"""### VQ-VAE observations

Test PSNR was {k512['final_psnr']:.2f} dB at K=512, {k256['final_psnr']:.2f} dB at K=256, and {k128['final_psnr']:.2f} dB at K=128. The K=256 model used {k256['codebook']['used']} of 256 codes; its perplexity was {k256['codebook']['perplexity']:.2f}. Its PixelCNN sample FID was {k256['fid_samples']:.2f}. Compare the code use, reconstruction FID, and generated image grids across K to assess codebook use and sample quality.
""")
    wg, gp = gan["wgan"], gan["wgan-gp"]
    replace_markdown(part_b, "### WGAN observations", f"""### WGAN observations

WGAN sample FID was {wg['final_fid']:.2f}, and WGAN-GP sample FID was {gp['final_fid']:.2f}. WGAN trained for {wg['train_sec']/60:.2f} minutes, and WGAN-GP trained for {gp['train_sec']/60:.2f} minutes. Compare the 100-image grids to assess recognizable shapes, sharpness, and variation. Reconstruction PSNR does not apply because these generators have no encoder.
""")
    nbformat.write(part_a, part_a_path)
    nbformat.write(part_b, part_b_path)
    print("Updated Part A and Part B observations from the full lab results")


if __name__ == "__main__":
    main()
