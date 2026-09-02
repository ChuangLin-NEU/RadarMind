# Acquisition protocol

## 三个必须能做的实验

当前 CI4R **做不到** 这三组对照。

### E1 Same Motion × Same Geometry × Different Radar

同一人、同一动作、同一距离/方位/朝向，24 与 77 同时采。
用来拆 **sensor effect**。

### E2 Same Motion × Different Geometry × Same Radar

同一雷达、同一人、同一动作，只改 distance / aspect / orientation。
用来拆 **geometry effect**，画单雷达 measurement manifold。

### E3 Same Geometry × Same Action × Different Subject

同一几何、同一动作，换人。
比较 subject variation 与 radar variation 谁大。

## 现场流程（单条）

1. 标定会话（当日一次）：空室、参考反射、站立。标记 `session_type=calibration`。
2. 摆几何：卷尺/地面标记对 distance 与 aspect；记录雷达高度与朝向。
3. 两雷达共触发或硬件同步；记下 `acquisition_id` 与统一时间戳。
4. 被试按动作表做一次；摄像机同时开。
5. 写完 metadata 再收下一条件。缺几何字段的条目标 `unusable`。

## 第一批规模（量级）

7 动作 × 3 距离 × 3 方位 × 8 人 × 3 次 × 2 雷达 ≈ 3000+ 同步条。
可先做 5 人、每条件 2 次作试点，但 grid 不要拆掉。

## 禁止

- 只存频谱图 PNG
- 先采完再按结果改 split
- 第一轮同时换房间、换遮挡、换很多新动作
- 用动作名代替几何量
