
import ipywidgets as widgets
from IPython.display import display,clear_output
import numpy as np 
import matplotlib.pyplot as plt

import scipy.stats
#### Methods for visualizing data for Proportional recovery rule ####
class PlotData:
    def __init__(self):
        self.x = None
        self.y = None 
        self.y_minus_x = None 
        self.ypred_minus_x = None 
        self.ypred = None 
        self.N = 100; 

    def generate_data(self, y_var=1., y_offs=0.): 
        self.x = np.random.randn(self.N)
        self.y = np.random.randn(self.N)*np.sqrt(y_var) + y_offs
        self.populate()

    def populate(self): 
        self.y_minus_x = self.y - self.x

        ### Pred outocme 
        slp,intc,rv,_,_ = scipy.stats.linregress(self.x, self.y)
        self.ypred = self.x*slp + intc 
        self.ypred_minus_x = self.ypred - self.x

        ### Outcome pred ###
        self.r_y_x = rv 

        ### Prop. recover 
        _,_,rv,_,_ = scipy.stats.linregress(self.x, self.y_minus_x)
        self.r_yminx_x = rv

class interative_viewer_plot(object):

    def __init__(self):

        ### Class for holding onto data 
        self.data = PlotData()

        self.out2 = widgets.Output()
        with self.out2:
            ### Plot the line bars 
            fig2, axes2 = plt.subplots(nrows = 2, figsize=(5, 9))
            self.format_ax2(axes2)
            plt.show(fig2)
        self.fig2 = fig2
        self.axes2 = axes2

        ### Get the childern: 
        self.children3 = [self.out2]

         ### Plot + Clear button 
        self.out1 = widgets.Output()
        with self.out1:
            ### Plot the line bars 
            fig1, axes1 = plt.subplots(nrows = 2, figsize=(5, 9))
            self.format_ax1(axes1)
            plt.show(fig1)
        self.fig = fig1 
        self.axes = axes1

        ### Clear button; 
        self.button=widgets.Button(description='Clear')
        self.button.on_click(self.clear_ax)
        self.children2 = [self.out1, self.button]

        #### Init conditions + time slider 
        self.make_data = widgets.interactive(self.plot_x, {'manual': True}, 
                                     log_var_y=widgets.FloatSlider(min=-2, max=2, step=.05),
                                     offs_y=widgets.FloatSlider(min=-2, max=2),
                                     style={'description_width': 'initial'})

        self.children1 = [self.make_data]

    def format_ax1(self, ax):
        for ia, axi in enumerate(ax): 
            axi.set_xlim([-.2, 1.2])
            if ia == 0: 
                axi.set_xticks([0, 1])
                axi.set_xticklabels(['Baseline (X)', 'Outcome (Y)'])
            elif ia == 1: 
                axi.set_xticks([0, 1])
                axi.set_xticklabels(['Baseline (X)', 'Recovery (Y-X)'])
            axi.set_ylabel('Scores')

    def format_ax2(self, ax):
        for ia, axi in enumerate(ax): 
            if ia == 0: 
                axi.set_xlabel('Baseline (X)')
                axi.set_ylabel('Outcome (Y)')
            elif ia == 1: 
                axi.set_xlabel('Baseline (X)')
                axi.set_ylabel('Recovery (Y-X)')

    def assemble_box(self):
        left = widgets.VBox(self.children1)
        right = widgets.VBox(self.children2)
        right_r = widgets.VBox(self.children3)
        self.box = widgets.HBox([left, right, right_r])

    def clear_ax(self, *args):
        with self.out1:
            clear_output(wait=True)
            self.data = PlotData()
            fig1, axes1 = plt.subplots(nrows = 2, figsize=(5, 9))
            self.format_ax1(axes1)
            plt.show(fig1)
        self.fig = fig1 
        self.axes = axes1

        with self.out2:
            clear_output(wait=True)
            fig2, axes2 = plt.subplots(nrows = 2, figsize=(5, 9))
            self.format_ax2(axes2)
            plt.show(fig2)
        self.fig2 = fig2
        self.axes2 = axes2

    def plot_x(self, log_var_y, offs_y, *args): 
        
        self.clear_ax()

        #### Generate data ###
        self.data.generate_data(y_var=10**log_var_y, y_offs=offs_y)
        
        #### Plot data ####
        with self.out1:
            clear_output(wait=True)
            self.format_ax1(self.axes)
            for n in range(self.data.N):
                self.axes[0].plot([0, 1], [self.data.x[n], self.data.y[n]], 'k.-')
                self.axes[0].set_title('VarY/varX=%.2f'%(np.var(self.data.y)/np.var(self.data.x)), color='b')
                self.axes[1].plot([0, 1], [self.data.x[n], self.data.y_minus_x[n]], 'k.-')
            self.fig.tight_layout()
            display(self.fig)

        ### Plot scatters ####
        with self.out2:
            clear_output(wait=True)
            self.format_ax2(self.axes)
            self.axes2[0].plot(self.data.x, self.data.y, 'k.')
            self.axes2[0].set_title('rv = %.3f'%self.data.r_y_x, color='b')
            self.axes2[0].plot(self.data.x, self.data.ypred, 'b-')

            self.axes2[1].plot(self.data.x, self.data.y_minus_x, 'k.')
            self.axes2[1].set_title('rv = %.3f'%self.data.r_yminx_x, color='b')
            self.axes2[1].plot(self.data.x, self.data.ypred_minus_x, 'b-')

            tmpx = np.arange(np.min(self.data.x), np.max(self.data.x))
            self.axes2[1].plot(tmpx, -1*tmpx, '--', color='gray')

            tmp = np.percentile(self.data.x, 95)
            self.axes2[1].text(tmp, tmp, 'Recovery =\n -X', color='gray')
            self.fig2.tight_layout()
            display(self.fig2)        


class interative_viewer_plot_PRR(object):

    def __init__(self):

        self.out2 = widgets.Output()
        with self.out2:
            ### Plot the line bars 
            fig2, axes2 = plt.subplots(ncols = 2, nrows = 2, figsize=(9, 9))
            self.format_ax2(axes2)
            plt.show(fig2)
        self.fig2 = fig2
        self.axes2 = axes2

        ### Get the childern: 
        self.children3 = [self.out2]


        #### Init conditions + time slider 
        self.make_data = widgets.interactive(self.plot_prr, {'manual': True}, 
                                     noise=widgets.IntSlider(min=0, max=50, step=1),
                                     ceiling=widgets.IntSlider(min=66, max=1000, step=1),
                                     style={'description_width': 'initial'})

        self.children1 = [self.make_data]

    def format_ax2(self, ax):
        for ia in range(2):
            for ib in range(2):
                if ib == 0: 
                    x = ''
                elif ib == 1: 
                    x = 'Shuff '

                ax[ib, ia].set_xlim([0, 66])

                ### columns 
                if ia == 0: 
                    ax[ib, ia].set_xlabel('Baseline (X)')
                    ax[ib, ia].set_ylabel('%sOutcome (Y)'%x)
                    ax[ib, ia].set_ylim([0, 100])

                elif ia == 1: 
                    ax[ib, ia].set_xlabel('Baseline (X)')
                    ax[ib, ia].set_ylabel('%sRecovery (Y-X)'%x)
                    ax[ib, ia].set_ylim([-66, 100])


    def assemble_box(self):
        left = widgets.VBox(self.children1)
        right_r = widgets.VBox(self.children3)
        self.box = widgets.HBox([left, right_r])

    def clear_ax(self, *args):
        with self.out2:
            clear_output(wait=True)
            fig2, axes2 = plt.subplots(ncols = 2, nrows = 2, figsize=(9, 9))
            self.format_ax2(axes2)
            plt.show(fig2)
        self.fig2 = fig2
        self.axes2 = axes2

    def plot_prr(self, noise, ceiling, *args): 
        
        self.clear_ax()

        #### Generate data ###
        #### Baseline scores (0-65) ###
        fm_scores = np.random.randint(0, 65, size=(100,))

        #### Compute recovery according to proportional recover curve
        outcomes = (66 - fm_scores)*0.7 + fm_scores
        outcomes = outcomes + noise*np.random.randn(len(fm_scores))
        outcomes = np.round(outcomes)
        outcomes[outcomes > ceiling] = ceiling

        ### Compute outcomes: 
        recovery = outcomes - fm_scores

        slp,intc,rv_out,_,_ = scipy.stats.linregress(fm_scores, outcomes)
        outcomes_pred = fm_scores*slp + intc; 

        slp, intc,rv_rec, _, _ = scipy.stats.linregress(fm_scores, recovery)
        recovery_pred = fm_scores*slp + intc 

        #### Shuffles shuffles ####
        ix_shuffle = np.random.permutation(len(fm_scores))
        outcomes_shuffle = outcomes[ix_shuffle]
        recovery_shuffle = outcomes_shuffle - fm_scores

        slp,intc,rv_out_shuff,_,_ = scipy.stats.linregress(fm_scores, outcomes_shuffle)
        outcomes_pred_shuffle = fm_scores*slp + intc; 

        slp, intc,rv_rec_shuff, _, _ = scipy.stats.linregress(fm_scores, recovery_shuffle)
        recovery_pred_shuffle = fm_scores*slp + intc 


        ### Plot scatters ####
        with self.out2:
            clear_output(wait=True)
            self.format_ax2(self.axes2)


            ### For the 
            self.axes2[0, 0].plot(fm_scores, outcomes, 'k.')
            self.axes2[0, 0].set_title('rv = %.3f, varY/varX=%.3f'%(rv_out, 
                np.var(outcomes)/np.var(fm_scores)), color='b')
            self.axes2[0, 0].plot(fm_scores, outcomes_pred, 'b-')

            self.axes2[0, 1].plot(fm_scores, recovery, 'k.')
            self.axes2[0, 1].set_title('rv = %.3f'%rv_rec, color='b')
            self.axes2[0, 1].plot(fm_scores, recovery_pred, 'b-')

            ### Shuffled 
            self.axes2[1, 0].plot(fm_scores, outcomes_shuffle, 'k.')
            self.axes2[1, 0].set_title('rv = %.3f, varY/varX=%.3f'%(rv_out_shuff, 
                np.var(outcomes_shuffle)/np.var(fm_scores)), color='b')
            self.axes2[1, 0].plot(fm_scores, outcomes_pred_shuffle, 'b-')

            self.axes2[1, 1].plot(fm_scores, recovery_shuffle, 'k.')
            self.axes2[1, 1].set_title('rv = %.3f'%rv_rec_shuff, color='b')
            self.axes2[1, 1].plot(fm_scores, recovery_pred_shuffle, 'b-')


            # tmpx = np.arange(66)
            # self.axes2[1].plot(tmpx, -1*tmpx, '--', color='gray')
            # self.axes2[1].text(50, -40, 'Recovery =\n -X', color='gray')
            self.fig2.tight_layout()
            display(self.fig2)        


# var_y,offs_y