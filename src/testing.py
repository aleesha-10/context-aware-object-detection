import os

print(" Checking for model files...")
print("=" * 50)

# Check for .h5 files
h5_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".h5"):
            h5_files.append(os.path.join(root, file))

if h5_files:
    print(" Found .h5 files:")
    for f in h5_files:
        print(f"  {f}")
else:
    print(" No .h5 files found")

print("\n" + "=" * 50)

# Check for .json files
json_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".json"):
            json_files.append(os.path.join(root, file))

if json_files:
    print(" Found .json files:")
    for f in json_files:
        print(f"  {f}")
else:
    print(" No .json files found")

print("\n" + "=" * 50)

# Check models directory
print("Checking models directory:")
models_path = "models"
if os.path.exists(models_path):
    print(f" {models_path}/ exists")
    for item in os.listdir(models_path):
        item_path = os.path.join(models_path, item)
        if os.path.isdir(item_path):
            print(f" {item}/")
            for subitem in os.listdir(item_path):
                print(f"  {subitem}")
        else:
            print(f" {item}")
else:
    print(f" {models_path}/ does not exist")

print("\n" + "=" * 50)
print(" If model files are missing, run:")
print("   python src/train_context_model.py")