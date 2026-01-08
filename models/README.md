# Model Weights

Large model weights are **not tracked directly** in this repository to keep the Git history clean and lightweight. Instead, they are distributed as **Release assets**.

---

## 📥 Download

1. Go to the [Releases](../../releases) page.
2. Select the release matching your code version (e.g., `v0.1.0`).
3. Download the following files into the `models/` folder:

   * `model.part1.rar`
   * `model.part2.rar`
   * `model.part3.rar`

---

## 🗜️ Recompose & Extract (Windows)

1. Ensure all `.rar` parts are in the same directory (`models/`).
2. Right-click `model.part1.rar` → **Extract Here** (using [WinRAR](https://www.win-rar.com/) or [7-Zip](https://www.7-zip.org/)).
3. The extractor will automatically join the parts and output the full weight file:

   ```
   models/weights/model.weights
   ```

---

## 🔒 Verify Integrity (optional)

After extraction, verify the SHA256 checksum matches the one provided in the Release notes:

```powershell
Get-FileHash models\weights\model.weights -Algorithm SHA256
```

Compare the output hash with the published checksum.

---

## 📂 Expected Structure After Extraction

```
models/
├─ config/
│  └─ darknet-yolov3.cfg
├─ weights/
│  └─ model.weights    # extracted from .rar parts
└─ classes.names
```

---

## 💡 Notes

* The `.rar` split ensures each file stays under GitHub’s 100 MB file size limit.
* If you prefer an alternative hosting solution (e.g., Hugging Face Hub), instructions will be added here in future releases.