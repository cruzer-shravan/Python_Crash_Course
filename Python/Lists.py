arr = [1, 2, 3, 4]

# Time complexity

arr[i]  # O(1)
arr.append(x)   # Amortized O(1) 
arr.insert(0, x)     # O(n) "All elements shift right"
del arr[i]    # O(n)    "Elements shift left"
x in arr    # O(n)  "Linear scan" - '''Interview trap'''

'''
Index access → O(1)
Append → O(1) amortized
Insert/delete → O(n)
Membership → O(n)
'''

# SPACE COMPLEXITY

''' 
Lists store:
Pointer array (contiguous)
Each element as a separate Python object

So memory is:   O(n)    

But with heavy overhead.

Each integer ≠ raw 4 bytes.
It is a full Python object (~28 bytes).
'''

[1,2,3] # far heavier than due internal behaviour [ ptr, ptr, ptr, ptr ]
np.array([1,2,3])

# BAD in ML (Bcoz runs in python interpreter - SLOW)
data = [1,2,3,... millions]

for x in data:
    x = x * 2


# GOOD in ML (10-100x faster)

data = np.array([...])
data = data * 2




# COMMON INTERVIEW TRAP

if x in large_list:
# SLOW because membership check is O(n).

if x in large_set:
# Set look up for O(1), Huge difference for large ML datasets.

'''How To Do Set Lookup'''

# Step 1 — Convert List to Set (Once!)
large_set = set(large_list)
# Now membership check:
if x in large_set:  # O(1) because sets use a hash table.

# IMPORTANT (Common Mistake)
if x in set(large_list):   # BAD - You’re rebuilding the set every time → O(n) each time.

# Correct pattern
large_set = set(large_list)   # Convert once

for x in queries:
    if x in large_set:
        ...

# ML-Relevant Example
bad_samples = [id1, id2, id3, ... millions]
for sample in dataset:
    if sample.id in bad_samples:   # O(n) each time ❌

'''If dataset has 1M rows and bad_samples has 1M rows:
Total worst case: O(n*2)
'''
# OPTIMIZED VERSION
bad_samples_set = set(bad_samples)

for sample in dataset:
    if sample.id in bad_samples_set:   # O(1)

# Now total O(n) - Massive difference

'''🧠 Why Set Is O(1)

Sets use:
Hashing
Buckets
Direct indexing via hash value
Instead of scanning linearly.
'''




'''LIST COMPREHENSION'''

[x*2 for x in arr] # O(n)


# Example: Training Loop Mistake
losses = []

for batch in dataset:
    loss = compute_loss(batch)
    losses.append(loss)

''' fOR BEST VERSION CHECK OUT CODE SNIPPETS'''


# MEMORY EXPLOSION in ML

data = []
for image in images:
    data.append(load_image(image))

