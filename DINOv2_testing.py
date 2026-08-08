import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import time

# ---------------------------------------------------------
# 1. Load the Heavyweight Model (The Translator)
# ---------------------------------------------------------
print("Loading DINOv2 (ViT-Large)... This will take more memory.")
# Stepping up to vitl14 (1024 dimensions)
# If your machine chokes on this, drop back to 'dinov2_vitb14' (Base, 768 dims)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.eval() 

# ---------------------------------------------------------
# 2. Aspect-Ratio Respecting Pipeline
# ---------------------------------------------------------
# DINOv2 demands height and width be multiples of 14. 
# 112 (Height) = 14 * 8
# 448 (Width)  = 14 * 32
# This 1:4 ratio prevents the "squashing" of long signatures.
transform = T.Compose([
    T.Resize((560, 504)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------
# 3. Extraction Function
# ---------------------------------------------------------
def get_signature_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Failed to load {image_path}: {e}")
        return None

    img_tensor = transform(img).unsqueeze(0) 

    with torch.no_grad(): 
        features = model(img_tensor)
    
    return features[0].numpy()

# ---------------------------------------------------------
# 4. The Real Stress Test
# ---------------------------------------------------------
if __name__ == "__main__":
    # You MUST test three files now, not two.
    file_real_1 = r"C:\Users\yashs\Downloads\Cheque Verification System\Signature Data\archive (3)\signature_ds_combined\0001\0001_01.jpg" # Baseline genuine
    file_real_2 = r"C:\Users\yashs\Downloads\Cheque Verification System\Signature Data\archive (3)\signature_ds_combined\0001\0001_02.jpg" # Second genuine (to test natural variance)
    file_fake = r"C:\Users\yashs\Downloads\Cheque Verification System\Signature Data\archive (3)\signature_ds_combined\0001_forg\0001F_01.jpg"     # The high-quality forgery

    print(f"\nProcessing signatures...")
    start_time = time.time()

    vec_real_1 = get_signature_embedding(file_real_1)
    vec_real_2 = get_signature_embedding(file_real_2)
    vec_fake = get_signature_embedding(file_fake)

    if all(v is not None for v in [vec_real_1, vec_real_2, vec_fake]):
        # The true test of your system
        dist_real_to_real = np.linalg.norm(vec_real_1 - vec_real_2)
        dist_real_to_fake = np.linalg.norm(vec_real_1 - vec_fake)
        
        print("\n--- RESULTS ---")
        print(f"Vector Dimensions: {vec_real_1.shape[0]}")
        print(f"Real vs. Real Distance: {dist_real_to_real:.4f} (This should be low)")
        print(f"Real vs. Fake Distance: {dist_real_to_fake:.4f} (This MUST be significantly higher)")
        print(f"Inference Time (Total for 3): {(time.time() - start_time):.2f} seconds")
        
        # The reality check
        margin = dist_real_to_fake - dist_real_to_real
        print(f"\nSeparation Margin: {margin:.4f}")
        
        if margin <= 0:
            print("🚨 FATAL: Your model thinks the forgery is closer to the baseline than your own second signature. The architecture fails here.")
        elif margin < 10.0:
            print("⚠️ WARNING: The margin is too tight. A slightly messy genuine signature will trigger a false rejection.")
        else:
            print("✅ PASS: You have a workable mathematical separation. Proceed to the Siamese network.")