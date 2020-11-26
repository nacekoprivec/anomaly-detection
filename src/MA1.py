import matplotlib.pyplot as plt
import numpy as np

#make the plots fancy
plt.style.use('dark_background') 

def warning_check(value, UL, LL):
#print warning if measurement is close to upper or lower limit
    value_normalized = 2*(value - (UL + LL)/2)/(UL - LL)
    if(value_normalized > 0.9):
        print('Warning: measurement close to upper limit.')
    if(value_normalized < -0.9):
        print('Warning: measurement close to lower limit.')
	

def live_plotter(x_vec,y1_data,lines,EMA, N,UL,LL, identifier='',pause_time=0.1):
    EMA = [i for i in EMA if i] 
    if lines[0]==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        # create a variable for lines so we can later update them
        lines[0], = ax.plot(x_vec,y1_data,'o',alpha=0.8) 
		
        EMA.append(np.average(y1_data[-N:]))
        sigma = np.std(y1_data[-N:])
        lines[1], = ax.plot(x_vec[-1], np.average(y1_data[-N:]), 'r-')
        lines[2], = ax.plot(x_vec[-1], EMA[-1] + sigma, 'b--')
        lines[3], = ax.plot(x_vec[-1], EMA[-1] - sigma, 'b--')
		
        ax.set_ylabel('Values')
        ax.set_title('{}'.format(identifier))

		#create histogram bins (range +- tolerance) -> currently cant handle events out of range
        tolerance = (UL - LL)/2
        bins = np.linspace(LL - tolerance, UL + tolerance, 30)
        bin_vals = np.zeros(len(bins))
        bin_vals[np.digitize([y1_data[-1]], bins)] = 1
        lines[4], = ax1.step(bins, bin_vals)
        
        ax1.set_ylabel('N of events')
        ax1.set_title('histogram'.format(identifier))
        plt.show()
	
    EMA.append(y1_data[-1] * (2/(N+1)) + EMA[-1]*(1-2/(N+1)))
    EMA = EMA[-len(x_vec):]
    sigma = np.std(y1_data[-N:])
    
	#updating lines for each new value
    lines[0].set_ydata(y1_data)
    lines[0].set_xdata(x_vec)
	
    lines[1].set_ydata(EMA)
    lines[1].set_xdata(x_vec[-len(EMA):])
    
    upper = lines[2].get_data()[1]
    upper = np.append(upper,EMA[-1] + sigma)
	
    lines[2].set_ydata(upper[-len(EMA):])
    lines[2].set_xdata(x_vec[-len(EMA):])

    lower = lines[3].get_data()[1]
    lower = np.append(lower,EMA[-1] - sigma)
	
    lines[3].set_ydata(lower[-len(EMA):])
    lines[3].set_xdata(x_vec[-len(EMA):])
	
    tolerance = (UL - LL)/2
    bins = np.linspace(LL - tolerance, UL + tolerance, 30)
    bin_vals = lines[4].get_data()[1]
    bin_vals[np.digitize([y1_data[-1]], bins)] += 1
    lines[4].set_ydata(bin_vals)
	
    # adjust limits if new data goes beyond bounds
    if (y1_data[-1])<=lines[0].axes.get_ylim()[0] or (y1_data[-1])>=lines[0].axes.get_ylim()[1]:
        plt.subplot(121).set_ylim([min(filter(lambda x: x is not None, y1_data)) - 1, max(filter(lambda x: x is not None, y1_data))+1])
    plt.subplot(121).set_xlim([min(filter(lambda x: x is not None, x_vec)) - 1, max(filter(lambda x: x is not None, x_vec))+1])

    if (max(bin_vals) >= lines[4].axes.get_ylim()[1]):
        plt.subplot(122).set_ylim([-1, max(bin_vals) + 1])
    
	
	
	# this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return lines, EMA
	
