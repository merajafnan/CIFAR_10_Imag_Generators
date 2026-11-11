"""Build Part C from full-run result files, or an honest preview from saved outputs."""

import argparse
import base64
import json
import tempfile
from pathlib import Path

import nbformat
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SOURCE = ROOT.parent.parent / "UDL" / "UDL_Assignment2_PartA_PartB.ipynb"
GROUP = [
    ("Meraj Hassan", "2025AE05085"),
    ("Himanshu Jha", "2025AG05420"),
    ("Sushil Kumar Behura", "2025AG05373"),
    ("Gundu Abhishek", "2025AG05701"),
    ("Nayudu Ganasai NagaSurya", "2025AG05591"),
]

# These figures are transcribed from the supplied notebook's saved full-run outputs.
# The final PDF uses the JSON files produced by the BITS lab rerun instead.
PREVIEW = {
    "vae": {
        "1": {"final_psnr": 20.60, "fid_samples": 145.99, "fid_recon": 145.10, "kl": 45.72, "sec_per_epoch": 4.14},
        "2": {"final_psnr": 19.35, "fid_samples": 175.33, "fid_recon": 180.67, "kl": 26.59, "sec_per_epoch": 4.01},
        "4": {"final_psnr": 18.18, "fid_samples": 220.65, "fid_recon": 222.82, "kl": 14.88, "sec_per_epoch": 4.02},
        "10": {"final_psnr": 16.86, "fid_samples": 272.19, "fid_recon": 274.81, "kl": 6.63, "sec_per_epoch": 4.43},
    },
    "vq": {
        "512": {"final_psnr": 31.757, "fid_samples": 81.807, "fid_recon": 10.086, "codebook": {"used": 512, "perplexity": 427.579}, "sec_per_epoch": 22.594, "sample_sec": 424.209},
        "256": {"final_psnr": 31.373, "fid_samples": 79.730, "fid_recon": 11.194, "codebook": {"used": 256, "perplexity": 216.832}, "sec_per_epoch": 18.127, "sample_sec": 410.720},
        "128": {"final_psnr": 30.542, "fid_samples": 79.853, "fid_recon": 14.532, "codebook": {"used": 128, "perplexity": 111.282}, "sec_per_epoch": 22.556, "sample_sec": 478.437},
    },
    "gan": {
        "wgan": {"final_fid": 80.49, "train_sec": 66.57 * 60, "sample_sec": 0.59},
        "wgan-gp": {"final_fid": 41.45, "train_sec": 134.57 * 60, "sample_sec": 0.58},
    },
}


def load_results(preview):
    if preview:
        return PREVIEW
    files = {
        "vae": RESULTS / "partA_vae.json",
        "vq": RESULTS / "partA_vqvae.json",
        "gan": RESULTS / "partB_wgan.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit("Run both full notebooks first. Missing: " + ", ".join(missing))
    data = {key: json.loads(path.read_text()) for key, path in files.items()}
    for key, values in (("vae", ("1", "2", "4", "10")), ("vq", ("512", "256", "128")), ("gan", ("wgan", "wgan-gp"))):
        if not all(value in data[key] for value in values):
            raise SystemExit(f"Incomplete {key} results; do not use QUICK mode for the final PDF")
    if any(len(data["vae"][beta]["psnr_curve"]) != 40 for beta in ("1", "2", "4", "10")):
        raise SystemExit("Part A VAE results are not from the full 40-epoch run")
    if any(len(data["vq"][k]["psnr_curve"]) != 30 or len(data["vq"][k]["pcnn_curve"]) != 20 for k in ("512", "256", "128")):
        raise SystemExit("Part A VQ-VAE and PixelCNN results are not from the full run")
    if any(data["gan"][mode]["fid_it"][-1] != 40000 for mode in ("wgan", "wgan-gp")):
        raise SystemExit("Part B GAN results are not from the full 40,000-iteration run")
    return data


def preview_images(directory):
    notebook = nbformat.read(SOURCE, as_version=4)
    specs = [(9, 1, "vae_preview.png"), (16, 1, "vq_preview.png"), (24, 0, "gan_preview.png")]
    result = []
    for cell_index, output_index, name in specs:
        data = notebook.cells[cell_index].outputs[output_index].data["image/png"]
        path = directory / name
        path.write_bytes(base64.b64decode(data))
        result.append(path)
    return result


def full_images():
    names = ("vae_beta1_100.png", "vqvae_k256_100.png", "wgan_100.png", "wgan-gp_100.png")
    paths = [RESULTS / "figs" / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("The 100-image grids are missing. Run both full notebooks first: " + ", ".join(missing))
    return paths


def make_table(rows, widths):
    table = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#20364b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#a9b2bb")),
    ]))
    return table


def scaled_image(path, max_width=7.0 * inch, max_height=8.1 * inch):
    with PILImage.open(path) as im:
        width, height = im.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width * scale, height * scale, hAlign="CENTER")


def build(preview=False):
    data = load_results(preview)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#20364b"), spaceAfter=12))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=5))
    styles["Heading2"].spaceBefore = 7
    styles["Heading2"].spaceAfter = 4
    styles.add(ParagraphStyle(name="CaptionSmall", parent=styles["BodySmall"], alignment=TA_CENTER, textColor=colors.HexColor("#4c5763")))
    filename = "PartC_Comparative_Analysis_PREVIEW.pdf" if preview else "PartC_Comparative_Analysis.pdf"
    target = ROOT / filename
    doc = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=42, rightMargin=42, topMargin=32, bottomMargin=32)
    story = []
    story.append(Paragraph("Part C Comparative Analysis of CIFAR-10 Image Generators", styles["ReportTitle"]))
    story.append(Paragraph("Group 9 | Unsupervised Deep Learning | S1-26_AIMLZG533", styles["BodySmall"]))
    if preview:
        story.append(Paragraph("PREVIEW: This document uses the saved outputs of the supplied notebook. The displayed grids contain 36 or 64 images. Rebuild after the full BITS Prayogshala run to include 100 images per required model.", styles["BodySmall"]))
    else:
        story.append(Paragraph("Final comparison from the full Part A and Part B notebook run. Each model page shows 100 generated images.", styles["BodySmall"]))
    story.append(Paragraph("Group members", styles["Heading2"]))
    story.append(make_table([["Name", "BITS ID"], *GROUP], [3.6 * inch, 2.0 * inch]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Method", styles["Heading2"]))
    story.append(Paragraph("All models use CIFAR-10 images scaled to [0, 1]. PSNR measures reconstruction of test images. Sample FID uses 10,000 generated images and the 10,000-image test set as its reference. GANs in this work have no encoder, so reconstruction PSNR does not apply to them.", styles["BodySmall"]))
    story.append(Paragraph("β-VAE results", styles["Heading2"]))
    vae_rows = [["β", "PSNR dB", "Sample FID", "Recon FID", "KL / image"]]
    for beta in ("1", "2", "4", "10"):
        v = data["vae"][beta]
        vae_rows.append([beta, f'{v["final_psnr"]:.2f}', f'{v["fid_samples"]:.2f}', f'{v["fid_recon"]:.2f}', f'{v["kl"]:.2f}'])
    story.append(make_table(vae_rows, [0.6 * inch, 1.15 * inch, 1.3 * inch, 1.3 * inch, 1.25 * inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("VQ-VAE with PixelCNN results", styles["Heading2"]))
    vq_rows = [["K", "PSNR dB", "Sample FID", "Recon FID", "Codes used", "Perplexity"]]
    for k in ("512", "256", "128"):
        v = data["vq"][k]
        vq_rows.append([k, f'{v["final_psnr"]:.2f}', f'{v["fid_samples"]:.2f}', f'{v["fid_recon"]:.2f}', f'{v["codebook"]["used"]}/{k}', f'{v["codebook"]["perplexity"]:.1f}'])
    story.append(make_table(vq_rows, [0.55 * inch, 1.0 * inch, 1.15 * inch, 1.15 * inch, 1.1 * inch, 0.95 * inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("GAN results", styles["Heading2"]))
    gan_rows = [["Model", "Sample FID", "Train time min", "Sampling time s / 10k"]]
    for name in ("wgan", "wgan-gp"):
        v = data["gan"][name]
        gan_rows.append([name.upper(), f'{v["final_fid"]:.2f}', f'{v["train_sec"] / 60:.2f}', f'{v["sample_sec"]:.2f}'])
    story.append(make_table(gan_rows, [1.2 * inch, 1.2 * inch, 1.5 * inch, 1.7 * inch]))

    with tempfile.TemporaryDirectory() as temp:
        images = preview_images(Path(temp)) if preview else full_images()
        labels = (
            [
                ("β-VAE sample preview", "The first panel shows 36 β=1 samples. Higher β panels are also shown for comparison."),
                ("VQ-VAE sample preview", "The middle panel shows 36 K=256 PixelCNN samples. K=512 and K=128 are also shown."),
                ("WGAN sample preview", "The left panel shows 64 WGAN samples. The right panel shows 64 WGAN-GP samples."),
            ]
            if preview
            else [
                ("β-VAE β=1: 100 generated images", "Random latent vectors were decoded by the β=1 VAE."),
                ("VQ-VAE K=256: 100 generated images", "PixelCNN sampled latent codes; the VQ-VAE decoder converted them to images."),
                ("WGAN: 100 generated images", "Random latent vectors were passed through the weight-clipped WGAN generator."),
                ("WGAN-GP: 100 generated images", "Random latent vectors were passed through the gradient-penalty generator."),
            ]
        )
        for path, (title, caption) in zip(images, labels):
            story.append(PageBreak())
            story.append(Paragraph(title, styles["ReportTitle"]))
            story.append(Paragraph(caption, styles["BodySmall"]))
            story.append(Spacer(1, 10))
            story.append(scaled_image(path))
            story.append(Spacer(1, 8))
            story.append(Paragraph("CIFAR-10, 32 × 32 RGB images", styles["CaptionSmall"]))

        story.append(PageBreak())
        story.append(Paragraph("Comparison and interpretation", styles["ReportTitle"]))
        v1, v10 = data["vae"]["1"], data["vae"]["10"]
        vq256 = data["vq"]["256"]
        wg, gp = data["gan"]["wgan"], data["gan"]["wgan-gp"]
        paragraphs = [
            f"<b>Reconstruction error.</b> VAE PSNR fell from {v1['final_psnr']:.2f} dB at β=1 to {v10['final_psnr']:.2f} dB at β=10. The KL term also fell. VQ-VAE K=256 reached {vq256['final_psnr']:.2f} dB. Its discrete latent representation preserved more visible detail than the tested VAE. Reconstruction PSNR is not defined for the two GANs in this implementation.",
            f"<b>Sampling quality.</b> Sample FID was {v1['fid_samples']:.2f} for VAE β=1, {vq256['fid_samples']:.2f} for VQ-VAE K=256, {wg['final_fid']:.2f} for WGAN, and {gp['final_fid']:.2f} for WGAN-GP. Lower FID indicates closer Inception feature statistics to the CIFAR-10 test images. The displayed WGAN-GP images show more recognizable shapes than the WGAN images. VAE samples are blurrier. PixelCNN samples are more detailed, but less exact than VQ-VAE reconstructions.",
            f"<b>Codebook quality.</b> The K=256 model used {vq256['codebook']['used']} of 256 codes. Perplexity was {vq256['codebook']['perplexity']:.1f}. Compare this use with the reconstruction and sample scores in the table.",
            f"<b>Speed.</b> The VAE run took about {v1['sec_per_epoch']:.1f} seconds per epoch at β=1. VQ-VAE K=256 took about {vq256['sec_per_epoch']:.1f} seconds per VQ epoch and {vq256['sample_sec']:.0f} seconds to sample 10,000 PixelCNN maps and decode them. WGAN took {wg['train_sec']/60:.1f} minutes to train; WGAN-GP took {gp['train_sec']/60:.1f} minutes. The methods used different training schedules, so these times describe computation cost rather than a controlled speed ranking.",
            "<b>Scope.</b> These results describe one CIFAR-10 run. FID is a feature-distance score; it does not establish that every generated image is realistic. The visual examples are needed to interpret the scores.",
        ]
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodySmall"]))
            story.append(Spacer(1, 7))

        doc.build(story)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Use saved notebook metrics and smaller embedded previews")
    build(parser.parse_args().preview)
