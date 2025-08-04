import shutil 
import os.path

if __name__ == '__main__':
   archived = shutil.make_archive('/home/griftj/phd/data/cad_aerial_25_', 'zip', '/home/griftj/phd/data/cad_aerial_25')

   if os.path.exists('/home/griftj/phd/data/cad_aerial_25_.zip'):
      print(archived) 
   else: 
      print("ZIP file not created")