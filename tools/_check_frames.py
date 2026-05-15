import csv
frames = set()
with open('bag_trial_1_20260512_190629_607/tf.csv') as f:
    for row in csv.DictReader(f):
        frames.add((row['frame_id'], row['child_frame_id']))
for p, c in sorted(frames):
    print(f'{p} -> {c}')
