import progressbar
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import xlwt
import random

def setup_seed(seed):
    #  Setting random seed
    print("[Setting]seed is [{}]".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_random_seed():
    seed = int(random.random() * 1000)
    setup_seed(seed)

class ModelUtils(object):

    def __init__(self):
        """模型工具类的资料初始化，如训练历史等等"""
        self.model = None  # 模型初始化
        self.loss_fc = None  # 损失函数初始化
        self.scheduler = None  # 学习率递减
        self.optimizer = None  # 优化器定义
        self.is_checkpoint = False  # 开启检查点
        self.checkpoint_key = None
        self.checkpoint_only_best = None
        self.checkpoint_out_dir = None
        # 是否是二分类
        self.is_two_category = False
        # 是否用半精度训练？
        self.is_half = False
        # 输出图像的路径设置
        self.output_image_dir = None
        # 输出历史相关信息
        self.is_checkpoint_output_image = False
        self.history = {
            "train_loss": list(),
            "test_loss": list(),
            "train_rank1": list(),
            "test_rank1": list(),
            "train_rank5": list(),
            "test_rank5": list(),
            # 下面都是二分类的指标！
            # 在分类中，当某个类别的重要性高于其他类别时，可以使用Precision和Recall多个比分类错误率更好的新指标。
            "train_precision": list(),  # "precision"
            "test_precision": list(),
            "train_recall": list(),  # "Recall/Sensitivity"
            "test_recall": list(),
            "train_f1": list(),  # "f1 score"
            "test_f1": list(),
            "train_specificity": list(),  # "specificity特异性"
            "test_specificity": list(),
            "train_FNR": list(),  # "漏诊率"
            "test_FNR": list(),
            "train_FPR": list(),  # "误诊率"
            "test_FPR": list(),
        }
        # 驱动程序加载
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def build(self, model):
        """模型工具类使用模型的建立，转移到device"""
        self.model = model
        # 转载到device中
        self.model = self.model.to(self.device)
        return self

    def compose(self, lr=1e-3, momentum=0.9, nesterov=True, weight_decay=1e-5,
                loss_fc=None, optimizer=None, scheduler=None, is_two_category=False,
                is_half=False
                ):
        """模型的优化方法、学习率衰减、权重、损失函数等等"""

        if loss_fc is None:
            self.loss_fc = nn.CrossEntropyLoss()
        else:
            self.loss_fc = loss_fc
        # 选择优化方法
        if optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                       momentum=momentum,
                                       nesterov=nesterov,
                                       weight_decay=weight_decay)
        else:
            self.optimizer = optimizer
        # 选择学习率递减
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        else:
            self.scheduler = scheduler
        # 是否是二分类任务
        if is_two_category:
            self.is_two_category = True
        # 是否是半精度训练
        if is_half:
            self.half()

        # Print Setting Environment
        setting_info = """model:{}\nloss_fc:{}\noptimizer:{}""".format(self.model.__class__.__name__,
                   self.loss_fc,
                   self.optimizer)
        print(setting_info)
        return self

    def fine_tuning(self, epochs=None, train_dataloader=None, test_dataloader=None, is_record_history=True):
        """微调网络"""
        model_layer = 0
        for param in self.model.parameters():
            model_layer += 1
        ft_list = [int(model_layer * 0.7), int(model_layer * 0.5), int(model_layer * 0.25), 0]
        for ft_times, s in enumerate(ft_list):
            i = 0
            for layer_param in self.model.parameters():
                if i >= s:
                    layer_param.requires_grad = True
                i += 1
            print("Begin No.[%d] fine-tuning !" % (ft_times + 1))
            self.train(epochs=epochs, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                       is_record_history=is_record_history)

    def half(self):
        """执行后，开启半精度训练，半精度训练对于30系显卡有加成，速度翻倍、显存减半！！！（可能存在的问题，据说可能会不收敛）"""
        if self.is_half is False:
            self.model.half()
            self.is_half = True

        else:
            print("[WARM]已经开启了半精度")
        # 模型半精度转换

        return self



    def train(self,
              epochs=None,
              train_dataloader=None,
              test_dataloader=None,
              is_record_history=True,
              is_softmax=False,
              ):
        """训练循环，要支持微调
        注意：如果模型最后有加softmax，请将is_softmax改为True，同时请不要用交叉熵代价函数
        """


        for epoch in range(epochs):
            training_loss = 0.0
            training_rank1 = 0
            training_rank5 = 0
            training_total = 0
            training_TP = 0
            training_FP = 0
            training_TN = 0
            training_FN = 0
            pbar = MyTqdm(epoch=epoch, maxval=len(train_dataloader))
            for i, (inputs, labels) in enumerate(train_dataloader):
                # 训练模式
                self.model.train()
                # GPU/CPU
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # 半精度
                if self.is_half:
                    inputs = inputs.half()
                    # labels = labels.half() # Label 值必须要是LONG类型
                self.optimizer.zero_grad()
                # foward
                outputs = self.model(inputs)
                # loss
                loss = self.loss_fc(outputs, labels)
                # loss求导，反向
                loss.backward()
                # 优化
                self.optimizer.step()
                training_loss += loss.item() / len(train_dataloader)
                rank1_this, rank5_this = self.rank(outputs, labels, is_softmax)
                training_rank1 += rank1_this
                training_rank5 += rank5_this
                training_total += labels.size(0)
                # 这里处理二分类指标
                # 检查是否有打开二分类测试
                if self.is_two_category:
                    pass
                    # 如果没有加softmax，这里要加一下。
                    if not is_softmax:
                        outputs = torch.nn.functional.softmax(outputs, dim=1)
                    # 这时output都是概率值了
                    prediction = torch.argsort(outputs, dim=1)
                    # 这时这里都是0和1了，注意！
                    for i, pre_labels in enumerate(prediction[:, -1]):
                        # TP    predict 和 label 同时为1
                        if labels[i] == 1 and pre_labels == 1:
                            training_TP += 1
                        elif labels[i] == 0 and pre_labels == 0:
                            training_TN += 1
                        elif labels[i] == 0 and pre_labels == 1:
                            training_FP += 1
                        elif labels[i] == 1 and pre_labels == 0:
                            training_FN += 1

                if self.is_two_category:
                    train_precision = training_TP / (training_TP + training_FP) \
                        if (training_TP + training_FP) != 0 else 0
                    train_recall = training_TP / (training_TP + training_FN) \
                        if (training_TP + training_FN) != 0 else 0
                    train_f1 = (2 * train_precision * train_recall) / (train_precision + train_recall) \
                        if (train_precision + train_recall) != 0 else 0
                    train_specificity = training_TN / (training_FP + training_TN) \
                        if (training_FP + training_TN) != 0 else 0
                    pbar.set_right_info(train_loss=loss.item(),
                                        train_acc=(training_TP + training_TN) / (
                                                    training_TP + training_TN + training_FN + training_FP),
                                        train_all_loss=training_loss,
                                        train_precision=train_precision,
                                        train_recall=train_recall,
                                        train_specificity=train_specificity,
                                        train_f1=train_f1,
                                        train_FNR=1-train_recall,
                                        train_FPR=1-train_specificity,
                                        )
                else:
                    pbar.set_right_info(train_loss=loss.item(),
                                        train_rank1=training_rank1 / training_total,
                                        train_rank5=training_rank5 / training_total,
                                        train_all_loss=training_loss,
                                        )
                # 測試
                pbar.update()
            else:
                self.scheduler.step()
                pbar.finish()
                if test_dataloader is not None:
                    self.test(epoch, test_dataloader, is_record_history)
            # 损失记录
            if is_record_history:
                self.history["train_loss"].append(training_loss)
                self.history["train_rank1"].append(training_rank1 / training_total)
                self.history["train_rank5"].append(training_rank5 / training_total)
                # 二分类的问题
                if self.is_two_category:
                    train_precision = training_TP / (training_TP + training_FP) \
                        if (training_TP + training_FP) != 0 else 0
                    train_recall = training_TP / (training_TP + training_FN) \
                        if (training_TP + training_FN) != 0 else 0
                    train_f1 = (2 * train_precision * train_recall) / (train_precision + train_recall) \
                        if (train_precision + train_recall) != 0 else 0
                    train_specificity = training_TN / (training_FP + training_TN) \
                        if (training_FP + training_TN) != 0 else 0
                    self.history["train_precision"].append(train_precision)
                    self.history["train_recall"].append(train_recall)
                    self.history["train_f1"].append(train_f1)
                    self.history['train_specificity'].append(train_specificity)
                    self.history['train_FNR'].append(1 - train_recall)
                    self.history['train_FPR'].append(1 - train_specificity)

            # 一个epoch的后续工作
            self.trained_work_queue()

    def test(self,
             epoch=None,
             test_dataloader=None,
             is_record_history=True,
             is_softmax=False,
             ):
        """验证一下怎么样"""
        pbar = MyTqdm(epoch=epoch, name="evaluate", maxval=len(test_dataloader))
        rank1_acc = 0
        rank5_acc = 0
        test_all_loss = 0
        total = 0
        # 二分类的指标
        test_TP = 0
        test_FP = 0
        test_TN = 0
        test_FN = 0
        # 评估模式
        self.model.eval()
        for i, (images_test, labels_test) in enumerate(test_dataloader):
            images_test = images_test.to(self.device)
            labels_test = labels_test.to(self.device)
            # 半精度
            if self.is_half:
                images_test = images_test.half()
                # labels_test = labels_test.half() # Label 需要 Long Type
            outputs_test = self.model(images_test)
            rank1_this, rank5_this = self.rank(outputs_test, labels_test, is_softmax)
            rank1_acc += rank1_this
            rank5_acc += rank5_this

            test_loss_this = self.loss_fc(outputs_test, labels_test).item()
            test_all_loss += test_loss_this / len(test_dataloader)
            total += labels_test.size(0)
            # 这里处理二分类指标
            # 检查是否有打开二分类测试
            if self.is_two_category:
                # 如果没有加softmax，这里要加一下。
                if not is_softmax:
                    outputs_test = torch.nn.functional.softmax(outputs_test, dim=1)
                # 这时output都是概率值了
                prediction = torch.argsort(outputs_test, dim=1)
                # 这时这里都是0和1了，注意！
                for i, pre_labels in enumerate(prediction[:, -1]):
                    # TP    predict 和 label 同时为1
                    if labels_test[i] == 1 and pre_labels == 1:
                        test_TP += 1
                    elif labels_test[i] == 0 and pre_labels == 0:
                        test_TN += 1
                    elif labels_test[i] == 0 and pre_labels == 1:
                        test_FP += 1
                    elif labels_test[i] == 1 and pre_labels == 0:
                        test_FN += 1

            pbar.update()
            if self.is_two_category:
                test_precision = test_TP / (test_TP + test_FP) \
                    if (test_TP + test_FP) != 0 else 0
                test_recall = test_TP / (test_TP + test_FN) \
                    if (test_TP + test_FN) != 0 else 0
                test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall) \
                    if (test_precision + test_recall) != 0 else 0
                test_specificity = test_TN / (test_FP + test_TN) \
                    if (test_FP + test_TN) != 0 else 0
                pbar.set_right_info(
                    test_loss=test_loss_this,
                    test_acc=(test_TP + test_TN) / (test_TP + test_TN + test_FN + test_FP),
                    test_all_loss=test_all_loss,
                    test_precision=test_precision,
                    test_recall=test_recall,
                    test_f1=test_f1,
                    test_specificity=test_specificity,
                    test_FNR=1 - test_recall,
                    test_FPR=1 - test_specificity,
                )
            else:
                pbar.set_right_info(
                    test_loss=test_loss_this,
                    test_rank1=rank1_acc / total,
                    test_rank5=rank5_acc / total,
                    test_all_loss=test_all_loss,
                )
        pbar.finish()
        if is_record_history:
            self.history["test_loss"].append(test_all_loss)
            self.history["test_rank1"].append(rank1_acc / total)
            self.history["test_rank5"].append(rank5_acc / total)
            # 二分类的问题
            if self.is_two_category:
                test_precision = test_TP / (test_TP + test_FP) \
                    if (test_TP + test_FP) != 0 else 0
                test_recall = test_TP / (test_TP + test_FN) \
                    if (test_TP + test_FN) != 0 else 0
                test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall) \
                    if (test_precision + test_recall) != 0 else 0
                test_specificity = test_TN / (test_FP + test_TN) \
                    if (test_FP + test_TN) != 0 else 0
                self.history["test_precision"].append(test_precision)
                self.history["test_recall"].append(test_recall)
                self.history["test_f1"].append(test_f1)
                self.history['test_specificity'].append(test_specificity)
                self.history["test_FNR"].append(1 - test_recall)
                self.history['test_FPR'].append(1 - test_specificity)

    def trained_work_queue(self):
        """训练过后的任务序列"""
        # 是否保存模型
        if self.is_checkpoint:
            self.checkpoint(self.checkpoint_key, self.checkpoint_only_best, self.checkpoint_out_dir)
        # 是否输出图像
        if self.is_checkpoint_output_image:
            self.output_trained_image(out_dir=self.output_image_dir)

    def open_checkpoint_output_image(self, out_dir="out/"):
        self.is_checkpoint_output_image = True
        self.output_image_dir = out_dir
        return self

    def open_checkpoint(self, key="test_rank1", only_best=True, out_dir="out/"):
        self.checkpoint_key = key
        self.checkpoint_only_best = only_best
        self.checkpoint_out_dir = out_dir
        self.is_checkpoint = True
        return self

    def close_checkpoint(self):
        self.is_checkpoint = False

    def checkpoint(self, key="test_rank1", only_best=True, out_dir="out/"):
        """检查点事件，手动触发"""
        if max(self.history[key]) == self.history[key][-1]:
            print("[info]正在保存模型...")
            if only_best:
                self.save(out_dir=out_dir)
            else:
                self.save("model_{}_{:.4f}.pt".format(key, self.history[key][-1]), out_dir=out_dir)

    def save(self, filename="model.pt", out_dir="out/"):
        if self.model is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            torch.save({'model': self.model.state_dict()}, out_dir + filename)
        else:
            print("[ERROR]:Please build a model!!!")

    def load(self, filepath=None):
        if filepath is not None:
            state_dict = torch.load(filepath)
            self.model.load_state_dict(state_dict['model'])
        else:
            print("[ERROR]:Load model occur a error!!!")

    def retrain(self):
        """继续训练模型
        废弃，原因load()+train()比这个更灵活。
        """
        pass

    @staticmethod
    def rank(output, label, is_softmax):
        """获取ACC值, label 采用class,output采用原始输出"""
        rank1 = 0
        rank5 = 0
        # 如果没有加softmax，这里要加一下。
        if not is_softmax:
            output = torch.nn.functional.softmax(output, dim=1)
        # print(output)
        if len(output) == len(label):
            prediction = torch.argsort(output, dim=1)
            for i, data in enumerate(prediction[:, -1]):
                if label[i].item() == data.item():
                    rank1 += 1
            for i, data in enumerate(prediction[:, -5:]):
                for k in data:
                    if label[i].item() == k.item():
                        rank5 += 1
                        break
        else:
            raise Exception()
        return rank1, rank5

    def output_trained_image(self, key_list=None, out_dir="out/"):
        """输出epochs全过程的相关指标变化,输出一个图片"""
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("[info]正在输出相关指标变化图...")
        plt.style.use("ggplot")
        if key_list is None:
            key_list = ['loss', 'rank1', 'rank5']
            if self.is_two_category:
                # 如果是二分类问题，还要加上其他东西的输出
                key_list.append('precision')
                key_list.append('recall')
                key_list.append('f1')
                key_list.append('specificity')
                key_list.append('FNR')
                key_list.append('FPR')
        for key in key_list:
            plt.figure()
            for name, value in self.history.items():
                if key in name:
                    plt.plot(np.arange(1, len(value) + 1), self.history[name], label=name)
            plt.title("%s Metrics" % key)
            plt.legend()
            try:
                plt.savefig(out_dir + '%s_Image.jpg' % key)
            except:
                plt.savefig(out_dir + '%s_Image.png' % key)
        return self

    def output_trained_excel(self, filename='All Metrics.xls', out_dir="out/"):
        """输出history(默认是模型训练的)全部指标到excel表格中"""
        print("[info]正在保存excel...")
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("All metrics")
        row = 0
        for key, value in self.history.items():
            sheet.write(0, row, key)
            for i in range(0, len(value)):
                sheet.write(i + 1, row, value[i])
            row += 1
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        workbook.save(out_dir + filename)
        return self


class MyProgressBar(object):
    def __init__(self, name="training", epoch=None, maxval=100):
        """maxval是输入的循环的次数，只能在 Python Console 下使用"""
        self.value = 0
        if epoch is None:
            self.widgets = [name + ': ', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer(),
                            ' ',
                            progressbar.ETA()]
        else:
            self.widgets = [name + ' epoch[%d]: ' % epoch, progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ',
                            progressbar.Timer(),
                            ' ', progressbar.ETA()]
        self.maxval = maxval
        self.pbar = progressbar.ProgressBar(widgets=self.widgets, maxval=self.maxval).start()

    def update(self):
        """更新一个数字"""
        self.value += 1
        self.pbar.update(self.value)

    def finish(self):
        self.pbar.finish()


class MyTqdm(object):

    def __init__(self, name="training", epoch=None, maxval=100):
        """maxval是输入的循环的次数，任何地方都可以使用"""
        self.maxval = maxval
        self.pbar = tqdm(total=self.maxval)
        if epoch is not None:
            self.set_left_info(name + " epoch[%d]" % epoch)
        else:
            self.set_left_info(name)

    def set_left_info(self, info=None):
        if info is not None:
            self.pbar.set_description(info)

    def set_right_info(self, **kwargs):
        if len(kwargs) > 0:
            self.pbar.set_postfix(**kwargs)

    def update(self):
        """对进度条增加特定的数值"""
        self.pbar.update(1)

    def finish(self):
        self.pbar.close()
