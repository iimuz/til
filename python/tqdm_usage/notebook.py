import sys

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def _main() -> None:
    """jupyter notebook と通常実行で tqdm を使い分けるための記述方法

    Note:
        jupyter環境で利用するためには、下記コマンドで対応環境もインストールする必要があります。
        ```sh
        $ pipenv install --dev jupyter jupyterlab ipywidgets
        $ pipenv run jupyter nbextension enable --py widgetsnbextension
        $ pipenv run jupyter labextension install @jupyter-widgets/jupyterlab-manager
        ```
    """
    for i in tqdm(range(10000)):
        i * i - i


if __name__ == "__main__":
    _main()
