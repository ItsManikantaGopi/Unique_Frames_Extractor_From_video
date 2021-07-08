import time
import save_file as save
# import imagecleaner 
import cv2 
import os
from image_compare import compare 
# from normalize import normalize 
from tqdm import tqdm 
#obr=cv2.ORB_create(nFeatures)

def extension_cutter(filename):
    """
    method will cut out the extension of the file name
    """
    temp=filename[::-1].find(".")
    string=""
    for i in filename:
        if i!=" ":
            string+=i
    return string[:-temp-1]
def video_reader(videoFile):
    """
    method to read the frames from the video
    and save those according to the requirement
    like non duplicate frames
    """
    vidcap = cv2.VideoCapture(videoFile)
    vidcap.set(cv2.CAP_PROP_FPS, int(30))
    success,prev = vidcap.read()
    base_img=prev
    seconds = 50
    fps = vidcap.get(cv2.CAP_PROP_FPS) 
    multiplier = fps * seconds
    save.save(prev,extension_cutter(file))
    i=1
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar=tqdm(total=frame_count)
    while success:
        pbar.update(i)
        frameId =0
        st=time.time()
        success, cur = vidcap.read()
        # cur2=normalize(cur,base_img)
        # if frameId % multiplier == 0:
        dir = extension_cutter(file)
        # print(dir)
        if compare(dir,cur):
            try:
                save.save(cur,dir)
                prev = cur
            except:
                continue
        
	    
    vidcap.release()
    # print("Complete")

	# imagecleaner(os.getcwd())


file = "3. Convolutions & Image Features.mp4"
# file="rbr.mp4"
video_reader(file)



"""	
def video_reader(file):
	print("video is reading")
	cam=cv2.VideoCapture(file)
	ret,prev=cam.read()
	print(ret)
	while (ret):
		ret,frame = cam 	.read()
		cur=frame
		#cv2.imshow("vid",frame)
		#cv2.waitKey()
		sf.save(prev,cur,extension_cutter(file))
		prev=cur
"""
# import time
# import multiprocessing 

# def basic_func(x):
#     if x == 0:
#         return 'zero'
#     elif x%2 == 0:
#         return 'even'
#     else:
#         return 'odd'

# def multiprocessing_func(x):
#     y = x*x
#     time.sleep(2)
#     print('{} squared results in a/an {} number'.format(x, basic_func(y)))
#     # print(time.time()-starttime)
    
# if __name__ == '__main__':
#     starttime = time.time()
#     processes = []
#     for i in range(0,10):
#         p = multiprocessing.Process(target=multiprocessing_func, args=(i,))
#         processes.append(p)
#         p.start()
        
#     for process in processes:
#         process.join()
        
#     print('That took {} seconds'.format(time.time() - starttime))

#     print("\n\n\n without multiprocessing\n\n")

#     st=time.time()
#     for i in range(0,10):
#     	multiprocessing_func(i)
#     print('That took {} seconds with out multi threads'.format(time.time() - st))

    
