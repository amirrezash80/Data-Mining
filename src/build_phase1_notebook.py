from pathlib import Path

import nbformat
from nbconvert.preprocessors import (
    ExecutePreprocessor,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

SOURCE_PATH = (
    NOTEBOOK_DIR
    / "phase1_eda.ipynb"
)

EXECUTED_PATH = (
    NOTEBOOK_DIR
    / "phase1_eda_executed.ipynb"
)


def markdown_cell(
    text: str,
):
    return nbformat.v4.new_markdown_cell(
        text
    )


def code_cell(
    code: str,
):
    return nbformat.v4.new_code_cell(
        code
    )


def build_notebook():
    notebook = nbformat.v4.new_notebook()

    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    }

    notebook["cells"] = [
        markdown_cell(
            "# Phase 1: Clustering-Oriented EDA\n\n"
            "Credit-card transaction clustering."
        ),
        markdown_cell(
            "## Research Question\n\n"
            "Do credit-card transactions form natural "
            "density-distinct subpopulations, and do "
            "small clusters or anomalous observations "
            "meaningfully overlap fraudulent transactions?"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, display\n"
            "\n"
            "ROOT = Path.cwd()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "\n"
            "REPORT_DIR = ROOT / 'reports' / 'phase1'\n"
            "FIGURE_DIR = ROOT / 'reports' / 'figures'\n"
            "DATA_DIR = ROOT / 'data' / 'processed'\n"
            "\n"
            "print('Project root:', ROOT)"
        ),
        markdown_cell(
            "## Data Profile"
        ),
        code_cell(
            "profile = pd.read_csv(\n"
            "    REPORT_DIR / 'clean_data_profile.csv'\n"
            ")\n"
            "profile"
        ),
        markdown_cell(
            "## Cleaning Decisions"
        ),
        code_cell(
            "with open(\n"
            "    REPORT_DIR / 'cleaning_decision_log.json',\n"
            "    encoding='utf-8',\n"
            ") as file:\n"
            "    cleaning_log = json.load(file)\n"
            "\n"
            "cleaning_log"
        ),
        markdown_cell(
            "## Extended EDA Summary"
        ),
        code_cell(
            "eda_summary = pd.read_csv(\n"
            "    REPORT_DIR / 'extended_eda_summary.csv'\n"
            ")\n"
            "eda_summary"
        ),
        markdown_cell(
            "## Scaling Comparison"
        ),
        code_cell(
            "display(Image(\n"
            "    filename=str(\n"
            "        FIGURE_DIR / 'phase1_scaling_before_after.png'\n"
            "    )\n"
            "))"
        ),
        markdown_cell(
            "## PCA Explained Variance"
        ),
        code_cell(
            "display(Image(\n"
            "    filename=str(\n"
            "        FIGURE_DIR / 'phase1_pca_explained_variance.png'\n"
            "    )\n"
            "))"
        ),
        markdown_cell(
            "## PCA Density Projection"
        ),
        code_cell(
            "display(Image(\n"
            "    filename=str(\n"
            "        FIGURE_DIR / 'phase1_pca_density.png'\n"
            "    )\n"
            "))"
        ),
        markdown_cell(
            "## UMAP Density Projection"
        ),
        code_cell(
            "display(Image(\n"
            "    filename=str(\n"
            "        FIGURE_DIR / 'phase1_umap_density.png'\n"
            "    )\n"
            "))"
        ),
        markdown_cell(
            "## Hopkins Statistic"
        ),
        code_cell(
            "hopkins = pd.read_csv(\n"
            "    REPORT_DIR / 'hopkins_results.csv'\n"
            ")\n"
            "\n"
            "hopkins.groupby('scaler')['hopkins'].agg(\n"
            "    ['mean', 'std', 'min', 'max']\n"
            ")"
        ),
        code_cell(
            "display(Image(\n"
            "    filename=str(\n"
            "        FIGURE_DIR / 'phase1_hopkins_results.png'\n"
            "    )\n"
            "))"
        ),
        markdown_cell(
            "## MST-Based VAT"
        ),
        code_cell(
            "display(Image(\n"
            "    filename=str(\n"
            "        FIGURE_DIR / 'phase1_vat_mst_heatmap.png'\n"
            "    )\n"
            "))"
        ),
        markdown_cell(
            "## Box Plots"
        ),
        code_cell(
            "display(Image(\n"
            "    filename=str(\n"
            "        FIGURE_DIR / 'phase1_feature_boxplots.png'\n"
            "    )\n"
            "))"
        ),
        markdown_cell(
            "## QQ Plots"
        ),
        code_cell(
            "display(Image(\n"
            "    filename=str(\n"
            "        FIGURE_DIR / 'phase1_feature_qqplots.png'\n"
            "    )\n"
            "))"
        ),
        markdown_cell(
            "## Scatter-Plot Matrix"
        ),
        code_cell(
            "display(Image(\n"
            "    filename=str(\n"
            "        FIGURE_DIR / 'phase1_feature_pairplot.png'\n"
            "    )\n"
            "))"
        ),
        markdown_cell(
            "## Significance-Masked Correlation"
        ),
        code_cell(
            "display(Image(\n"
            "    filename=str(\n"
            "        FIGURE_DIR / "
            "'phase1_significance_masked_correlation.png'\n"
            "    )\n"
            "))"
        ),
        markdown_cell(
            "## Phase 1 Deliverables\n\n"
            "- Reproducible ingestion and preprocessing\n"
            "- Cleaning decision log\n"
            "- Feature engineering\n"
            "- Standard and robust scaling comparison\n"
            "- PCA and UMAP\n"
            "- Hopkins clustering tendency\n"
            "- MST-based VAT\n"
            "- Clustering-oriented EDA\n"
            "- PDF report"
        ),
    ]

    return notebook


def main() -> None:
    NOTEBOOK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    notebook = build_notebook()

    nbformat.write(
        notebook,
        SOURCE_PATH,
    )

    executor = ExecutePreprocessor(
        timeout=1200,
        kernel_name="python3",
    )

    executor.preprocess(
        notebook,
        {
            "metadata": {
                "path": str(ROOT)
            }
        },
    )

    nbformat.write(
        notebook,
        EXECUTED_PATH,
    )

    print(
        f"Notebook created: {SOURCE_PATH}"
    )

    print(
        f"Executed notebook created: "
        f"{EXECUTED_PATH}"
    )


if __name__ == "__main__":
    main()