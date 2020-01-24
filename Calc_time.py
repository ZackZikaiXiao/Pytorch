#In linux:
import time
start = time.time()
run_function()
end = time.time()
print('Running time: %s Seconds'%(end-start))

#In windows:
import time
start = time.clock()
run_function()
end = time.clock()
print (str(end-start))
print('Running time: %s Seconds'%(end-start))
