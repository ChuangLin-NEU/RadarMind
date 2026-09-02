# Cross-environment feasibility

只检查当前本地已有数据集的元数据。没有下载新数据，也没有生成 environment label。

## A. 哪个 dataset 有真实环境变量？

当前三个本地雷达集都没有经过核验的环境变量。

| 数据集 | 结论 |
|---|---|
| CI4R | 无 room / environment / distance / angle 字段。目录是动作名。文件名前缀不是官方 subject，更不是环境。 |
| mRI | JSON 只有 `subject` 与 `labels[action]=[start,end]`（Frame #）。README / videolabels 没有 room、距离、角度。`environment_id` 在适配器里是 UNKNOWN。 |
| RadHAR | 官方只有 Train/Test 动作文件夹。无 subject，无 environment。 |

## B. 哪些字段有官方证据？

- CI4R：动作来自目录名；77 GHz 是 IQ；10/24/77 官方 PNG 有频段目录。没有环境表。
- mRI：Dryad README + `videolabels/{subject}.json` + radar CSV 表头 `Frame #,# Obj,X,Y,Z,Doppler,Intensity,Time`。官方身份字段是 subject。
- RadHAR：GitHub 仓库的 `Data/Train|Test/{action}/*.txt`。官方划分是这两个文件夹。

以上都不是环境字段。

## C. 当前能否直接定义 cross-environment split？

不能。没有可验证的 environment / room / session-site 标签，不能按环境划分 train/test。

## D. 最缺什么？

缺一份带出处的环境标注：房间、雷达摆放、距离或视角，并且能对到现有每条 recording。没有这份表之前，`cross_environment` 保持 `available=false`。

这是当前数据范围下的结论，不是实验失败。
