"""把热成像伪彩亮度换算成小鼠实际体温的工具集。

工作流（在 `alignment/` 之后）：

    1. alignment.dlc sample 产出 *_thermal_intensity.csv   (灰度强度)
    2. thermometry.fit         拟合 intensity → 温度的标定
    3. thermometry.apply       逐帧逐 bodypart 换算成温度，并聚合得到体温
    4. thermometry.summary     汇总每段视频的统计
    5. thermometry.overlay     可选：在 thermal 视频上叠加温度数字
"""
