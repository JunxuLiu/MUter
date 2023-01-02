from select import select
from turtle import title, width
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from argument import argument
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from itertools import cycle

class Ploter:

    def __init__(self, repeat_times):
        """
        1) For Clean Accuracy, we plot the picture let the line in a small region.
        2) For Perturbed Accuracy ,we plot line picture, let line show the effect of MUter.
        3) For Distance, we plot as accurate as possible.
        4) For Time ...
        """

        self.root_path = '../data/Result'
        self.repeat_times = repeat_times
        self.dataset_list = [
            'binaryMnist', #0
            'covtype', #1
            # 'epsilon', #2
            # 'madelon', #3
            # 'phishing', #4
            # 'splice', #5
        ]

        self.adv_type_list = [
            'FGSM', #0
            'PGD', #1
        ]

        self.model_list = [
            'logistic', #0
            'ridge', #1
        ]

        self.remove_type_map = [
            'one_step_single_point', #0
            'one_step_batch_points',  #1
            'multiple_steps_single_point', #2 
            'multiple_steps_batch_points' #3
        ]

        self.remove_method_list = [
            'retrain',
            'MUter', 
            'Newton',
            'Influence',
            'Fisher',
            'Newton_delta',
            'influence_delta',
            'Fisher_delta',
        ]

        self.time_remove_method_list = [
            'MUter', 
            'retrain',
            'SISA',
        ]
    
        self.type2_dict = {
            'binaryMnist': [1, 2, 3, 4, 5, 120, 240, 360, 480, 600],
            'madelon': [1, 2, 3, 4, 5, 20, 40, 60, 80, 100],
            'phishing': [1, 2, 3, 4, 5, 100, 200, 300, 400, 500],
            'splice': [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            'covtype': [1, 2, 3, 4, 5, 5000, 10000, 15000, 20000, 25000],
            'epsilon': [1, 2, 3, 4, 5, 4000, 8000, 12000, 16000, 20000],
        }

        self.type2_dict_random1 = {
            'binaryMnist': [1, 2, 3, 4, 5, 120, 240, 360, 480, 597.3],
            'madelon': [1, 2, 3, 4, 5, 20, 40, 60, 80, 100],
            'phishing': [1, 2, 3, 4, 5, 100, 200, 300, 400, 500],
            'splice': [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            'covtype': [1, 2, 3, 4, 5, 5000, 10000, 15000, 20000, 24892.83],
            'epsilon': [1, 2, 3, 4, 5, 4000, 8000, 12000, 16000, 19948.21],
        }

        self.type2_dict_random2 = {
            'binaryMnist': [1, 2, 3, 4, 5, 120, 240, 360, 480, 601.43],
            'madelon': [1, 2, 3, 4, 5, 20, 40, 60, 80, 100],
            'phishing': [1, 2, 3, 4, 5, 100, 200, 300, 400, 500],
            'splice': [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            'covtype': [1, 2, 3, 4, 5, 5000, 10000, 15000, 20000, 24982.92],
            'epsilon': [1, 2, 3, 4, 5, 4000, 8000, 12000, 16000, 20003.4],
        }

        self.type3_dict = {
            'binaryMnist': [i * 120 for i in range(1, 11)],
            'madelon': [i * 20 for i in range(1, 11)],
            'phishing': [i * 100 for i in range(1, 11)],
            'splice': [i * 10 for i in range(1, 11)],
            'covtype': [i * 5000 for i in range(1, 11)],
            'epsilon': [i * 4000 for i in range(1, 11)], 
        }

    def Load(self, dataset_id, metrics, repeat_times, remove_type, remove_adv_type, remove_model):

        if metrics != 'time':
            remove_methods = self.remove_method_list
        else:
            remove_methods = self.time_remove_method_list
        
        datas_lists = []
        delete_num_lists = []
        methods_lists = []
        models_lists = []      

        print(len(remove_methods)*repeat_times)

        for method in remove_methods:
            if method == 'retrain' and metrics == 'distance':
                continue

            for dex in range(repeat_times):
                path = os.path.join(
                    self.root_path, 
                    '{}/{}/{}_adv_{}_model_{}_method_{}_experiment{}.npy'.format(
                        self.dataset_list[dataset_id], 
                        self.remove_type_map[remove_type], 
                        metrics, 
                        self.adv_type_list[remove_adv_type], 
                        self.model_list[remove_model], 
                        method, 
                        dex
                    )
                )
                print(path) 
                arr = np.load(path)
                
                datas_lists.append(arr)
                print(len(datas_lists), len(datas_lists[-1]))
                methods_list = [method for i in range(1, len(arr)+1)]
                methods_lists.append(methods_list)
                
                if metrics == 'time':
                    models_list = [self.model_list[remove_model].capitalize() for i in range(1, len(arr)+1)]
                    models_lists.append(models_list)
                
                if remove_type == 2:
                    delete_num_list = self.type2_dict[self.dataset_list[dataset_id]]
                    delete_num_lists.append(delete_num_list)

                if remove_type == 3:
                    delete_num_list = self.type3_dict[self.dataset_list[dataset_id]]
                    delete_num_lists.append(delete_num_list)

        print(len(datas_lists)*len(datas_lists[0]), len(methods_lists), len(delete_num_lists))
        datas_lists = np.concatenate([datas_list for datas_list in datas_lists])
        methods_lists = np.concatenate([methods_list for methods_list in methods_lists])  

        if metrics == 'time':
            models_lists = np.concatenate([models_list for models_list in models_lists])

        if remove_type in [2, 3]:
            delete_num_lists = np.concatenate([delete_num_list for delete_num_list in delete_num_lists]) 
        
        print(len(datas_lists), len(methods_lists), len(delete_num_lists))

        dicter = {}

        if remove_type in [2, 3]:
            dicter['remove_numbers'] = delete_num_lists
        if metrics == 'time':
            dicter['model'] = models_lists
        if remove_type == 0:
            dicter['remove_type'] = [0 for i in range(len(datas_lists))]
        
        dicter['method'] = methods_lists
        dicter[metrics] = datas_lists
        print(len(dicter['method']), len(dicter[metrics]), len(dicter['remove_numbers']))

        df = pd.DataFrame.from_dict(dicter)
        return df

    def trans_dataset_name(self, name):

        dicter = {
            'binaryMnist': 'MNIST-b',
            'madelon': 'Madelon',
            'phishing': 'Phishing',
            'splice': 'Splice',
            'covtype': 'Covtype',
            'epsilon': 'Epsilon',
        }

        return dicter[name]

    # def plot(
    #     self,
    #     save_name,
    #     metrics,
    #     info = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)], # point what dataset and size that we what to plot.
    # ):
    #     fig = plt.figure(figsize=(24, 4 * len(info)))
    #     sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
    #     sns.set_palette(palette=sns.color_palette('bright'))

    #     patches = None
    #     line_handles, line_labels = None, None

    #     for index in range(len(info) * 4):

    #         sub_info = info[index // 4]
    #         ax = plt.subplot(len(info), 4, index + 1)

    #         df = self.Load(
    #             dataset_id=sub_info[0], 
    #             metrics=metrics,
    #             repeat_times=self.repeat_times,
    #             remove_type=index % 4, 
    #             remove_adv_type=sub_info[1],
    #             remove_model=sub_info[2],
    #         )

    #         # index % 4 == 0 remove type == 0
    #         if index % 4 in [0, 1]:
    #             temp = sns.barplot(
    #                 x='method', 
    #                 y=metrics, 
    #                 data=df,
    #                 ax=ax, 
    #             )

    #             if metrics == 'acc':
    #                 plt.ylabel('Clean Accuracy')
    #                 temp.set_ylim(self.set_acc_param(df))
    #             elif metrics == 'distance':
    #                 plt.ylabel('Distance')
    #             elif metrics == 'perturbed_acc':
    #                 plt.ylabel('Perturbed Accuracy')


    #             plt.title('{}  {}  {}'.format(
    #                 self.model_list[sub_info[2]].capitalize(),
    #                 self.adv_type_list[sub_info[1]], 
    #                 self.trans_dataset_name(self.dataset_list[sub_info[0]]), 
    #                 )
    #             )
    #             patches = [
    #                 matplotlib.patches.Patch(color=sns.color_palette()[i], label=t) for i,t in enumerate(t.get_text() for t in temp.get_xticklabels())
    #             ]
    #             plt.xlabel('')
    #             plt.xticks([])
    #             temp.set_title(ax.get_title(), size=14)
    #             temp.set_ylabel(ax.get_ylabel(), size=14)

    #             if metrics == 'perturbed_acc':

    #                 plt.axhline(y=df.loc[df['method'] == 'retrain'][metrics].mean(), linestyle='--', color='r')
    #                 temp.set_ylim(self.set_param(dataset_id=sub_info[0], remove_type=index % 4, remove_model=sub_info[2]))

    #         elif index % 4 in [2, 3]:
    #             if metrics == 'acc':
    #                 plt.ylabel('Clean Accuracy')
    #             elif metrics == 'distance':
    #                 plt.ylabel('Distance')
    #             elif metrics == 'perturbed_acc':
    #                 plt.ylabel('Perturbed Accuracy')

    #             self_marker = ['o', 'v', '^', 's', '*', 'X', 'd', 'P']

    #             if metrics == 'distance':
    #                 self_marker = ['o', '^', 's', '*', 'X', 'd', 'P']

    #             temp = sns.lineplot(
    #                 x='remove_numbers', 
    #                 y=metrics, 
    #                 data=df, 
    #                 style='method', 
    #                 markers=self_marker, 
    #                 hue='method', 
    #                 linewidth=1.5,
    #                 ax=ax,
    #             )
    #             if metrics == 'acc':
    #                 temp.set_ylim(self.set_acc_param(df))
    #             line_handles, line_labels = ax.get_legend_handles_labels()
    #             temp.legend_.remove()

    #             plt.xlabel('Removal Numbers')
    #             plt.title('{}  {}  {}'.format(
    #                 self.model_list[sub_info[2]].capitalize(),
    #                 self.adv_type_list[sub_info[1]], 
    #                 self.trans_dataset_name(self.dataset_list[sub_info[0]]), 
    #                 )
    #             )
    #             temp.set_title(ax.get_title(), size=14)
    #             temp.set_ylabel(ax.get_ylabel(), size=14)
    #             temp.set_xlabel(ax.get_xlabel(), size=14)
    
    #             if index % 4 == 2 and metrics in ['distance', 'perturbed_acc']:
    #                 if metrics == 'distance':
    #                     axins = inset_axes(ax, width='40%', height='30%', loc='upper left', bbox_to_anchor=(0.12, -0.02, 1, 1), bbox_transform=ax.transAxes)
    #                 elif metrics == 'perturbed_acc':
    #                     axins = inset_axes(ax, width='40%', height='30%', loc='lower left', bbox_to_anchor=(0.1, 0.08, 1, 1), bbox_transform=ax.transAxes)
    #                 sub_temp = sns.lineplot(
    #                     x='remove_numbers', 
    #                     y=metrics, 
    #                     data=df, 
    #                     style='method', 
    #                     markers=self_marker, 
    #                     hue='method', 
    #                     linewidth=1.5,
    #                     ax=axins,
    #                 )
    #                 # 调整子坐标系的显示范围
    #                 axins.set_xlabel('')
    #                 axins.set_ylabel('')
    #                 axins.set_xticks([1, 2, 3, 4 , 5])
    #                 axins.set_xlim(0.8, 5.2)

    #                 axins.set_ylim(self.set_axin_ylim(df, metrics))
    #                 sub_temp.legend_.remove()
    #                 if metrics == 'distance':
    #                     mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='k', lw=1)
    #                 elif metrics == 'perturbed_acc':
    #                     mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec='k', lw=1)


    #     plt.subplots_adjust(wspace=0.3, hspace=0.3)
    #     fig.legend(handles=patches, bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=18)
    #     fig.legend(line_handles, line_labels, bbox_to_anchor=(0.91, 0.98), ncol=4, fontsize=18)
    #     plt.savefig('{}.pdf'.format(save_name), dpi=800, bbox_inches='tight', pad_inches=0.2)
    #     plt.close()

    def cvpr_plot(
        self,
        save_name,
        metrics,
        adv_type,
        remove_type=2,
        remove_model=0,
        datasetId=0, # point what dataset and size that we what to plot.
    ):
        """
        only show SinSR, SuccSR mode.
        """
        fig = plt.figure(figsize=(18, 4))
        sns.set_style('darkgrid')
        # sns.set_style('whitegrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})
        # sns.set_palette(palette=sns.color_palette('bright'))

        patches = None
        bar_label = None
        line_handles, line_labels = None, None

        dataset_id = datasetId
        ax = plt.subplot(1, 4, 2)
        
        df = self.Load(
            dataset_id=dataset_id, 
            metrics=metrics,
            repeat_times=self.repeat_times,
            remove_type=2,
            remove_adv_type=adv_type,
            remove_model=remove_model,
        )

        if metrics == 'acc':
            plt.ylabel('Clean Accuracy')
        elif metrics == 'distance':
            plt.ylabel('Distance')
        elif metrics == 'perturbed_acc':
            plt.ylabel('Perturbed Accuracy')

        # self_marker = ['o', 'v', '^', 's', '*', 'X', 'd', 'P']
        self_marker = ['o', 'o', 'v', '^', 's', 'v', '^', 's']

        if metrics == 'distance':
            # self_marker = ['o', '^', 's', '*', 'X', 'd', 'P']
            self_marker = ['o', 'v', '^', 's', 'v', '^', 's']

        if metrics in ['acc', 'perturbed_acc']:
            palette=['#FF7C00', '#023EFF', '#1AC938', '#E8000B', '#8B2BE2', '#9F4800', '#F14CC1', '#A3A3A3']
        else:
            palette=['#023EFF', '#1AC938', '#E8000B', '#8B2BE2', '#9F4800', '#F14CC1', '#A3A3A3']  

        temp = sns.lineplot(
            x='remove_numbers', 
            y=metrics, 
            data=df, 
            style='method', 
            markers=self_marker, 
            hue='method', 
            linewidth=2,
            ax=ax,
            ci=None,
            dashes=False,
            palette=palette,
        )
        if metrics == 'acc':
            temp.set_ylim(self.set_acc_param(df))
        line_handles, line_labels = ax.get_legend_handles_labels()
        temp.legend_.remove()

        plt.xlabel('Removal Numbers')
        plt.title('{}  {}  {}'.format(
            self.model_list[remove_model].capitalize(),
            self.adv_type_list[adv_type], 
            self.trans_dataset_name(self.dataset_list[datasetId]), 
            )
        )
        temp.set_title(ax.get_title(), size=14)
        temp.set_ylabel(ax.get_ylabel(), size=14)
        temp.set_xlabel(ax.get_xlabel(), size=14)

        if metrics in ['distance', 'perturbed_acc']:
            if metrics == 'distance':
                axins = inset_axes(ax, width='40%', height='30%', loc='upper left', bbox_to_anchor=(0.12, -0.02, 1, 1), bbox_transform=ax.transAxes)
            elif metrics == 'perturbed_acc':
                axins = inset_axes(ax, width='40%', height='30%', loc='lower left', bbox_to_anchor=(0.1, 0.08, 1, 1), bbox_transform=ax.transAxes)

            sub_temp = sns.lineplot(
                x='remove_numbers', 
                y=metrics, 
                data=df, 
                style='method', 
                markers=self_marker, 
                hue='method', 
                linewidth=1.5,
                ax=axins,
                ci=None,
                dashes=False,
                palette=palette
            )
            # 调整子坐标系的显示范围
            axins.set_xlabel('')
            axins.set_ylabel('')
            axins.set_xticks([1, 2, 3, 4, 5])
            axins.set_xlim(0.8, 5.2)

            axins.set_ylim(self.set_axin_ylim(df, metrics))
            sub_temp.legend_.remove()
            
            # draw a bbox of the region of the inset axes in the parent axes and
            # connecting lines between the bbox and the inset axes area
            # if metrics == 'distance':
            #     mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='k', lw=1)
            # elif metrics == 'perturbed_acc':
            #     # mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec='k', lw=1)
            #     mark_inset(ax, axins, fc="none", ec='k', lw=1)

        if metrics != 'distance':
            order_list = [0, 1, 2, 5, 3, 6, 4, 7]
        else:
            order_list = [1, 4, 2, 5, 3, 6, 0]

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        # print([patches[i] for i in order_list])
        # print([bar_label[i] for i in order_list])
        # fig.legend(handles=[patches[i] for i in order_list], labels=[bar_label[i] for i in order_list], bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=12)

        # fig.legend(bar_handles, bar_labels, bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=16)
        fig.legend([line_handles[i] for i in order_list], [line_labels[i] for i in order_list], bbox_to_anchor=(0.9, 0.995), ncol=4, fontsize=12)
        plt.savefig('{}.pdf'.format(save_name), dpi=2000, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    def set_axin_ylim(self, df, metrics):

        if metrics == 'distance':
            sub_df = df[df.remove_numbers < 6]
            minValue = sub_df[metrics].min()
            maxValue = sub_df[metrics].max()
            addingValue = 0.1 * (maxValue - minValue)
            return (minValue - addingValue, maxValue + addingValue)
        
        elif metrics == 'perturbed_acc':
            sub_df = df[df.remove_numbers < 6]
            minValue = sub_df[metrics].min()
            maxValue = sub_df[metrics].max()
            addingValue = 0.1 * (maxValue - minValue)
            return (minValue - addingValue, maxValue + addingValue)
    
    def set_bar_range(self, df, metrics):
        
        if metrics == 'perturbed_acc':
            minValue = df[metrics].min()
            maxValue = df[metrics].max()
            addingValue = 0.1 * (maxValue - minValue)
            return (minValue - addingValue, maxValue + addingValue)
        elif metrics == 'distance':
            minValue = df[metrics].min()
            maxValue = df[metrics].max()
            addingValue = (maxValue - minValue)
            return (max(minValue - addingValue, 0), maxValue + addingValue)
        elif metrics == 'acc':
            minValue = df[metrics].min()
            maxValue = df[metrics].max()
            addingValue =  15 * (maxValue - minValue)
            return (minValue - addingValue, maxValue + addingValue)          

    def set_param(self, dataset_id, remove_type, remove_model):

        type0_perturbed_acc_dict = {
            'binaryMnist': {'logistic':(0.87, 0.93), 'ridge':(0.85, 0.93)}, 
            'covtype': {'logistic':(0.5, 0.68), 'ridge':(0.5, 0.68)},
            'splice': {'logistic':(0.55, 0.75), 'ridge':(0.55, 0.75)}, 
            'phishing': {'logistic':(0.6, 0.9), 'ridge':(0.6, 0.9)}, 
            'madelon': {'logistic':(0.3, 0.75), 'ridge':(0.3, 0.75)}, 
            'epsilon': {'logistic':(0.6, 0.75), 'ridge':(0.6, 0.75)},
        }
        type1_perturbed_acc_dict = {
            'binaryMnist': {'logistic':(0.2, 0.95), 'ridge':(0.2, 0.95)},
            'covtype': {'logistic':(0.1, 0.85), 'ridge':(0.1, 0.85)}, 
            'splice': {'logistic':(0.55, 0.75), 'ridge':(0.55, 0.75)}, 
            'phishing': {'logistic':(0.5, 0.85), 'ridge':(0.6, 0.9)}, 
            'madelon': {'logistic':(0.3, 0.75), 'ridge':(0.3, 0.75)}, 
            'epsilon': {'logistic':(0.1, 0.85), 'ridge':(0.1, 0.85)},
        }
        
        if remove_type == 0:
            return type0_perturbed_acc_dict[self.dataset_list[dataset_id]][self.model_list[remove_model]]
        else:
            return type1_perturbed_acc_dict[self.dataset_list[dataset_id]][self.model_list[remove_model]]

    def set_acc_param(self, df):
        
        minValue = df['acc'].min()
        maxValue = df['acc'].max()

        if maxValue - minValue < 0.03:
            return (minValue - 0.05, min(maxValue + 0.05, 1.0))
        elif maxValue - minValue < 0.06:
            return (minValue - 0.08, min(maxValue + 0.08, 1.0))
        else:
            return (minValue - 0.1, min(maxValue + 0.1, 1.0))

    # def plot_time(
    #     self,
    #     save_name,
    #     dataset_id_list = [0, ],
    # ):
    #     """
    #     plot the special dataset time, the form is line, 
    #     for example line: [FGSM+remove_type0, PGD+type0, FGSM+type1, PGD+type1], 
    #     every col include logistic and ridge
    #     """

    #     remove_type_name = ['SinSR', 'SinBR']

    #     fig = plt.figure(figsize=(24, 4 * len(dataset_id_list)))
    #     sns.set_style('darkgrid', {'axes.linewidth': 2, 'axes.edgecolor':'black'})

    #     for index in range(len(dataset_id_list) * 4):
            
    #         ax = plt.subplot(len(dataset_id_list), 4, index + 1)

    #         df1 = self.Load(
    #             dataset_id=dataset_id_list[index // 4], 
    #             metrics='time',
    #             repeat_times=self.repeat_times,
    #             remove_type=((index % 4) // 2),  # map to 0, 0, 1, 1, 0, 0, 1, 1,...
    #             remove_adv_type=(index % 2),
    #             remove_model=0, # Logistic
    #         )

    #         df2 = self.Load(
    #             dataset_id=dataset_id_list[index // 4], 
    #             metrics='time',
    #             repeat_times=self.repeat_times,
    #             remove_type=((index % 4) // 2),  # map to 0, 0, 1, 1, 0, 0, 1, 1,...
    #             remove_adv_type=(index % 2),
    #             remove_model=1, #Ridge
    #         )

    #         df = pd.concat([df1, df2], ignore_index=True)
    #         sns.set_palette(palette=sns.color_palette(['orange', 'blue', 'red'], desat=0.6))
    #         temp = sns.barplot(x='model', y='time', data=df, hue='method', ax=ax)

    #         plt.annotate(
    #             '{:.4f}s'.format(df.loc[(df['method'] == 'MUter') & (df['model'] == 'Logistic')]['time'].mean()), 
    #             xy=(0.12, 0.01), 
    #             xytext=(0.05, 0.1),
    #             xycoords='axes fraction',
    #             arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color="black"),
    #         )

    #         plt.annotate(
    #             '{:.4f}s'.format(df.loc[(df['method'] == 'MUter') & (df['model'] == 'Ridge')]['time'].mean()), 
    #             xy=(0.62, 0.01), 
    #             xytext=(0.55, 0.1),
    #             xycoords='axes fraction',
    #             arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color="black"),
    #         )

    #         plt.xlabel('')
    #         plt.ylabel('Removal Time (s)')
    #         plt.title(
    #             '{}  {}  {}'.format(self.adv_type_list[index % 2], remove_type_name[(index % 4) // 2], self.trans_dataset_name(self.dataset_list[dataset_id_list[index // 4]]))                
    #         )
    #         ax.set_xticklabels(plt.xticks()[1], size=16)
    #         ax.set_ylabel(ax.get_ylabel(), size=16)
    #         plt.legend(loc='upper left')
    #         plt.setp(ax.get_legend().get_texts(), fontsize='16')
    #         plt.setp(ax.get_legend().get_title(), fontsize='16')


    #     plt.subplots_adjust(wspace=0.3, hspace=0.3)
    #     plt.savefig('{}.pdf'.format(save_name), dpi=800, bbox_inches='tight', pad_inches=0.2)
    #     plt.close()

    def cvpr_time_table(
            self,
            save_name,
            adv_type=1,
            datasetId_list=[0, 1, 2],
        ):

        def timeInfo_to_string(method, dataset, df1, df2, df3, df4, df21=None, df41=None):

            dicter1 = {
                0: 1,
                2: 5,
                4: self.type2_dict_random1[dataset][9],
            }
            dicter2 = {
                0: 1,
                2: 5,
                4: self.type2_dict_random2[dataset][9],               
            }
            
            if method == 'retrain':
                ssr = '&{} '.format(method)
                ssr += '&{:.4f} '.format(df1.loc[df1['method'] == method]['time'].mean())
                # Logistic SuccSR
                for index in [0, 2, 4]:
                    remove_number = self.type2_dict[dataset][index]
                    ssr += '&{:.4f} '.format(df2.loc[(df2['method'] == method) & (df2['remove_numbers'] == remove_number)]['time'].mean())
                # Ridge SinSR 
                ssr += '&{:.4f} '.format(df3.loc[df3['method'] == method]['time'].mean())
                # Ridge SuccSR
                for index in [0, 2, 4]:
                    remove_number = self.type2_dict[dataset][index]
                    if index != 4:
                        ssr += '&{:.4f} '.format(df4.loc[(df4['method'] == method) & (df4['remove_numbers'] == remove_number)]['time'].mean())
                    else:
                        ssr += '&{:.4f}\\\\'.format(df4.loc[(df4['method'] == method) & (df4['remove_numbers'] == remove_number)]['time'].mean())
                return ssr   

            elif method == 'SISA':
                ssr = '&{} '.format(method)
                # Logistic SinSR 
                temp1 = df2.loc[(df2['method'] == method) & (df2['remove_numbers'] == self.type2_dict[dataset][0])]['time'].mean()
                ssr += '&{:.4f} '.format(df1.loc[df1['method'] == method]['time'].mean()*0.1 + temp1*0.9)
                # Logistic SuccSR
                for index in [0, 2]:
                    remove_number = self.type2_dict[dataset][index]
                    ssr += '&{:.4f} '.format(df2.loc[(df2['method'] == method) & (df2['remove_numbers'] == remove_number)]['time'].mean())
                ssr += '&{:.4f} '.format(df21.loc[df21['method'] == method]['time'].mean())
                # Ridge SinSR 
                temp2 = df4.loc[(df4['method'] == method) & (df4['remove_numbers'] == self.type2_dict[dataset][0])]['time'].mean()
                ssr += '&{:.4f} '.format(df3.loc[df3['method'] == method]['time'].mean()*0.1 + temp2*0.9)
                # Ridge SuccSR
                for index in [0, 2]:
                    remove_number = self.type2_dict[dataset][index]
                    ssr += '&{:.4f} '.format(df4.loc[(df4['method'] == method) & (df4['remove_numbers'] == remove_number)]['time'].mean())
                ssr += '&{:.4f}\\\\'.format(df41.loc[df41['method'] == method]['time'].mean())               
                return ssr

            else:
                ssr = '&{} '.format(method)
                # Logistic SinSR 
                temp1 = df2.loc[(df2['method'] == method) & (df2['remove_numbers'] == self.type2_dict[dataset][0])]['time'].mean()
                ssr += '&\\textbf{'
                ssr += '{:.4f}'.format(df1.loc[df1['method'] == method]['time'].mean()*0.1 + temp1*0.9)
                ssr += '} '
                # Logistic SuccSR
                for index in [0, 2, 4]:
                    remove_number = self.type2_dict[dataset][index]
                    ssr += '&\\textbf{'
                    ssr += '{:.4f}'.format(df2.loc[(df2['method'] == method) & (df2['remove_numbers'] == remove_number)]['time'].mean() * dicter1[index])
                    ssr += '} '
                # Ridge SinSR 
                temp2 = df4.loc[(df4['method'] == method) & (df4['remove_numbers'] == self.type2_dict[dataset][0])]['time'].mean()
                ssr += '&\\textbf{'
                ssr += '{:.4f} '.format(df3.loc[df3['method'] == method]['time'].mean()*0.1 + temp2*0.9)
                ssr += '} '
                # Ridge SuccSR
                for index in [0, 2, 4]:
                    remove_number = self.type2_dict[dataset][index]
                    if index != 4:
                        ssr += '&\\textbf{'
                        ssr += '{:.4f} '.format(df4.loc[(df4['method'] == method) & (df4['remove_numbers'] == remove_number)]['time'].mean() * dicter2[index])
                        ssr += '} '
                    else:
                        ssr += '&\\textbf{'
                        ssr += '{:.4f}'.format(df4.loc[(df4['method'] == method) & (df4['remove_numbers'] == remove_number)]['time'].mean() * dicter2[index])
                        ssr += '}\\\\'
                return ssr

        str = ''

        for index in range(len(datasetId_list)):
            
            df1 = self.Load( #logistc SinSR
                dataset_id=datasetId_list[index], 
                metrics='time',
                repeat_times=self.repeat_times,
                remove_type=0,
                remove_adv_type=adv_type,
                remove_model=0,
            )

            df2 = self.Load( #logistc SuccSR
                dataset_id=datasetId_list[index], 
                metrics='time',
                repeat_times=self.repeat_times,
                remove_type=2,
                remove_adv_type=adv_type,
                remove_model=0, 
            )

            df21 = self.Load(
                dataset_id=datasetId_list[index],
                metrics='time',
                repeat_times=self.repeat_times,
                remove_type=1,
                remove_adv_type=adv_type,
                remove_model=0,
            )
            
            df3 = self.Load( #Ridge SinSR
                dataset_id=datasetId_list[index], 
                metrics='time',
                repeat_times=self.repeat_times,
                remove_type=0,
                remove_adv_type=adv_type,
                remove_model=1,
            )

            df4 = self.Load( #Ridge SuccSR
                dataset_id=datasetId_list[index], 
                metrics='time',
                repeat_times=self.repeat_times,
                remove_type=2,
                remove_adv_type=adv_type,
                remove_model=1, 
            )

            df41 = self.Load(
                dataset_id=datasetId_list[index],
                metrics='time',
                repeat_times=self.repeat_times,
                remove_type=1,
                remove_adv_type=adv_type,
                remove_model=1,
            )

            str += '\\multirow{3}{*}{\\textbf{'
            str += '{}'.format(self.dataset_list[datasetId_list[index]])
            str += '}} '
            str += timeInfo_to_string('MUter', self.dataset_list[datasetId_list[index]], df1, df2, df3, df4)
            str += '\n'
            str += timeInfo_to_string('SISA', self.dataset_list[datasetId_list[index]], df1, df2, df3, df4, df21, df41)
            str += '\n'
            str += timeInfo_to_string('retrain', self.dataset_list[datasetId_list[index]], df1, df2, df3, df4)
            str += '\n'
            str +='\\hline\n'

        return str        

if __name__ == "__main__":

    ploter = Ploter(repeat_times=5)
    # for i in range(1):    
        # ploter.plot(save_name='distance_{}'.format(i), metrics='distance', info=[(i, 0, 0), [i, 0, 1], [i, 1, 0], [i, 1, 1]])
        # ploter.plot(save_name='acc_{}'.format(i), metrics='acc', info=[(i, 0, 0), [i, 0, 1], [i, 1, 0], [i, 1, 1]])
        # ploter.plot(save_name='perturbed_acc_{}'.format(i), metrics='perturbed_acc', info=[(i, 0, 0), [i, 0, 1], [i, 1, 0], [i, 1, 1]])\
    
    ploter.cvpr_plot(save_name='Context_metrics_distance', metrics='distance', adv_type=1, datasetId=0)
    # ploter.cvpr_plot(save_name='Context_metrics_acc', metrics='acc', adv_type=1, datasetId=0)
    # ploter.cvpr_plot(save_name='Context_metrics_perturbed_acc', metrics='perturbed_acc', adv_type=1, datasetId=0)

    # ploter.cvpr_plot(save_name='Appendix_metrics_distance', metrics='distance', adv_type=0, datasetId_list=0)
    # ploter.cvpr_plot(save_name='Appendix_metrics_acc', metrics='acc', adv_type=0, datasetId_list=0)
    # ploter.cvpr_plot(save_name='Appendix_metrics_perturbed_acc', metrics='perturbed_acc', adv_type=0, datasetId_list=0)

    # ploter.plot_time(save_name='time_picture', dataset_id_list=[0, 1, 2, 3, 4, 5])

    # print(ploter.cvpr_time_table(save_name='hehe', adv_type=0))
