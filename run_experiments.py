import argparse
import os
import time
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Yazlab Zaman Serisi Anomali Tespiti Deneyleri")
    parser.add_argument("--fast", action="store_true", help="Deneyleri kisa surede bitirmek icin hizli mod (Sadece 1 Seed ve kisa Epoch)")
    args = parser.parse_args()
    
    print("="*60)
    print("  YAZLAB PROJESI - DENEY MODULU (RUN EXPERIMENTS)")
    print("="*60)
    
    if args.fast:
        print("[BILGI] FAST (Hizli) mod aktif edildi! Deney yaklasik 1-2 dakika surecektir.")
        # Hizli mod için çevre değişkeni ayarlıyoruz
        os.environ["YAZLAB_FAST_MODE"] = "1"
    else:
        print("[BILGI] UZUN (Kapsamli) mod aktif edildi! (Arkadaslarinin dedigi 20-40 dk suren mod)")
        print(">> Tum Seed'ler (5 adet), tam Epoch egitimleri ve kapsamli capraz testler yapilacak...")
        os.environ["YAZLAB_FAST_MODE"] = "0"
        
    start_time = time.time()
    
    # main.py'yi alt islem olarak calistir
    print("\n>>> Deney Boru Hatti (Pipeline) Baslatiliyor...\n")
    try:
        subprocess.run(["python", "main.py"], check=True)
    except subprocess.CalledProcessError:
        print("\n[HATA] Deneyler sirasinda bir kod hatasi olustu!")
        return
        
    end_time = time.time()
    
    print("\n" + "="*60)
    print(f"  TUM DENEYLER BASARIYLA TAMAMLANDI! (Gecen Sure: {(end_time - start_time)/60:.2f} dakika)")
    print("  -> Skor tablolari 'logs/' klasorune eklendi.")
    print("  -> Grafikler 'figures/' klasorunde guncellendi.")
    print("="*60)

if __name__ == "__main__":
    main()
