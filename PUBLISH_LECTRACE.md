# Publishing the LecTrace demo website

The built site in `docs/` is completely self-contained: all asset paths are
relative and the figure is inlined as base64, so it works at a project subpath
such as `https://<user>.github.io/<repo>/`.

**Do not use a GitHub Actions build for this project.** The CI runner has no GPU,
no MedMNIST download and no SimCLR checkpoints, so executing the trace there
would fail. Publish the pre-built `docs/` folder instead.

## Steps

1. Create a GitHub repository, e.g. `md6329-self-supervision`.
2. From the project folder:

   ```
   git init
   git add .
   git commit -m "MD6329 project: self-supervised pretraining under label scarcity"
   git branch -M main
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```

3. On GitHub: **Settings -> Pages**.
   - Source: **Deploy from a branch**
   - Branch: **main**, folder: **/docs**
   - Save.
4. Wait about a minute, then open `https://<user>.github.io/<repo>/`.
   That URL is the "LecTrace Script/Demo Website" link for the cover page.

## If you later change the presentation

```
python -c "from lectrace.cli import main; main()" build lectrace_presentation.py -o docs --no-incremental
echo > docs/.nojekyll
git add docs && git commit -m "rebuild lectrace site" && git push
```

## Large files

`results/preds/` holds 210 small .npz files and `ckpt/` holds 12 checkpoints
(~45 MB each). The checkpoints are NOT needed to view the site; add this to
`.gitignore` if the push is too large:

```
ckpt/
data/
```
