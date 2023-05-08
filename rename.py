import os
folder_path = 'random_64_128'
m=12968;
for i, file_name in enumerate(os.listdir(folder_path)):
    if file_name.endswith('.npz'):
        old_path = os.path.join(folder_path, file_name)
        new_file_name = '{}.npz'.format(m, i)
        new_path = os.path.join(folder_path, new_file_name)
        os.rename(old_path, new_path)
        m=m+1

