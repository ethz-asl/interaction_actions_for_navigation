import numpy as np
from matplotlib import pyplot as plt

rew_hist_filepath="/tmp/instant_rewards.txt"
def watch_rewplot(mode=0, path=rew_hist_filepath, pause_duration=1):
    """
    mode 0 is active mode, the axes limits get periodically updated
    mode 1 is passive mode, move freely """
    try:
        first = True
        while True:
            try:
                rewplot = np.loadtxt(path, delimiter=",").T
                cumsum = np.cumsum(rewplot, axis=1)
            except:
                plt.clf()
                plt.pause(pause_duration)
                continue
            # reset sum at each crash / arrival
            final = rewplot * 1.
            final[np.abs(final) > 10] = -cumsum[np.abs(final) > 10]
            final_cumsum = np.cumsum(final, axis=1)
            # plot
            plt.ion()
            if mode == 0 or first: # Active
                fig = plt.gcf()
                fig.clear()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                lines1 = []
                lines2 = []
                for curve in rewplot:
                    line1, = ax1.plot(curve) # Returns a tuple of line objects, thus the comma
                    lines1.append(line1)
                for curve in cumsum:
                    line2, = ax2.plot(curve)
                    lines2.append(line2)
                plt.show()
            if mode == 1: # Passive
                for line1, curve in zip(lines1, rewplot):
                    line1.set_data(np.arange(len(curve)), curve)
                for line2, curve in zip(lines2, cumsum):
                    line2.set_data(np.arange(len(curve)), curve)
                fig.canvas.draw()
                fig.canvas.flush_events()
            plt.pause(pause_duration)
            first = False
    except KeyboardInterrupt:
        mode += 1
        if mode > 1:
            plt.close()
            raise KeyboardInterrupt
        print("Switching to passive mode. Graph will still update, but panning and zoom is now free.")
        print("Ctrl-C again to close and quit")
        watch_rewplot(mode=1, path=path, pause_duration=pause_duration)

if __name__=="__main__":
    watch_rewplot()
