from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
need = ["yolo26n.pt", "baseline_best_val_f1.pt", "hufep_best_val_f1.pt"]

print("BASE:", BASE)
print("MODELS:", MODEL_DIR)
print("\n필수 파일 확인")

missing = []
for name in need:
    p = MODEL_DIR / name
    if p.exists():
        print("OK  ", name, f"{p.stat().st_size/1024/1024:.1f} MB")
    else:
        print("MISS", name)
        missing.append(name)

print("\n브라우저 실행: http://127.0.0.1:8000")
if missing:
    print("\n누락:", ", ".join(missing))
else:
    print("\n모든 필수 파일이 있습니다.")
