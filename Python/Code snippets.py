# Example: Training Loop Mistake
losses = []

for batch in dataset:
    loss = compute_loss(batch)
    losses.append(loss)

'''Case 1 — You Only Need Average Loss (Best Practice)'''

# Best Version (Memory Efficient)

total_loss = 0.0
num_batches = 0

for batch in dataset:
    loss = compute_loss(batch)
    total_loss += loss
    num_batches += 1

avg_loss = total_loss / num_batches

'''Case 2 — PyTorch Training Loop (Correct & Safe)'''

# Best Version for PyTorch

total_loss = 0.0

for batch in dataset:
    loss = compute_loss(batch)
    total_loss += loss.item()   # Detach from graph

avg_loss = total_loss / len(dataset)


'''Case 3 — Vectorized Loss (Even Better)'''

# Instead of

for batch in dataset:
    loss = compute_loss(batch)

# Do:
all_predictions = model(full_dataset)
loss = loss_fn(all_predictions, targets)


'''Case 4 — If You Actually Need All Losses (Clean Version)'''

losses = [compute_loss(batch) for batch in dataset]


'''Best Professional Training Loop (Full Version) - PRODUCTION READY '''

model.train()
total_loss = 0.0

for batch in dataloader:
    optimizer.zero_grad()
    
    outputs = model(batch["x"])
    loss = loss_fn(outputs, batch["y"])
    
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()

avg_loss = total_loss / len(dataloader)









'''MEMORY EXPLOSION'''

# The BAD Way (Memory Explosion) - You just tried to load ~1TB into RAM (200_000 images x 5MB per image)
data = []

for image_path in image_paths:
    img = load_image(image_path)   # loads full image into RAM
    data.append(img)


# 1️⃣ Generator Example (Lazy Loading) - Generator loads one item at a time

def image_generator(image_paths):
    for path in image_paths:
        yield load_image(path)   # yield instead of return

# Usage - Memory complexity - O(1)
for img in image_generator(image_paths):
    process(img)

'''
What happens?
    Only ONE image in memory at a time
    After processing → memory released
    RAM stays constant
'''

# 2️⃣ Generator With Streaming Batches - batch properly

def batch_generator(image_paths, batch_size):
    batch = []
    
    for path in image_paths:
        img = load_image(path)
        batch.append(img)
        
        if len(batch) == batch_size:
            yield batch
            batch = []
    
    if batch:   # leftover batch
        yield batch

# Usage - Memory complexity - O(n)

for batch in batch_generator(image_paths, batch_size=32):
    train_step(batch)

'''
Now memory is:  O(batch_size)
                Not O(dataset_size).

🧠 Why This Is Powerful

Instead of storing 100,000 images:
You store only 32 at a time.
Huge difference.
'''


# 3️⃣ Professional Version — PyTorch DataLoader 

# Step 1 - Custom Dataset

from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = load_image(path)
        return image
    
'''
Notice:
    It does NOT preload images
    Loads only when requested
'''

# Step 2 — DataLoader (Automatic Batching + Streaming)

from torch.utils.data import DataLoader

dataset = ImageDataset(image_paths)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
for batch in dataloader:
    outputs = model(batch)
    loss = loss_fn(outputs)
    loss.backward()
    optimizer.step()


'''
What DataLoader Does Internally
    Loads data lazily
    Batches automatically
    Parallel loading (num_workers)
    Avoids RAM explosion
    Streams batches to GPU

Memory usage:   O(batch_size)
'''