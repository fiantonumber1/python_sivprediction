import os
import gzip
import shutil
import glob

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
gz_files = sorted(glob.glob(os.path.join(folder, "*.gz")))

if not gz_files:
    print("Tidak ada file .gz ditemukan.")
else:
    print(f"Ditemukan {len(gz_files)} file .gz\n")
    for gz_path in gz_files:
        basename = os.path.basename(gz_path)

        # Tentukan nama output dan jumlah layer gz
        if basename.endswith('.csv.gz.gz'):
            out_name = basename[:-len('.gz.gz')] + ''   # strip satu .gz dulu
            # out_name = DDMMYYYY.csv
            out_name = basename.replace('.csv.gz.gz', '.csv')
            layers = 2
        elif basename.endswith('.csv.gz'):
            out_name = basename.replace('.csv.gz', '.csv')
            layers = 1
        else:
            print(f"  Skip (ekstensi tidak dikenal): {basename}")
            continue

        out_path = os.path.join(folder, out_name)

        if os.path.exists(out_path):
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"  Sudah ada (skip): {out_name}  ({size_mb:.1f} MB)")
            continue

        print(f"  Ekstrak ({layers}x): {basename:35s}  ->  {out_name}")
        try:
            if layers == 1:
                with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            else:
                # Layer 1: .gz.gz -> .gz (di memory)
                import io
                with gzip.open(gz_path, 'rb') as f1:
                    inner = f1.read()
                # Layer 2: .gz -> csv
                with gzip.open(io.BytesIO(inner), 'rb') as f2, open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f2, f_out)

            size_mb = os.path.getsize(out_path) / 1e6
            print(f"    Selesai ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"    ERROR: {e}")
            if os.path.exists(out_path):
                os.remove(out_path)

print("\nEkstraksi selesai!")
print(f"Total CSV di folder: {len(glob.glob(os.path.join(folder, '*.csv')))}")
