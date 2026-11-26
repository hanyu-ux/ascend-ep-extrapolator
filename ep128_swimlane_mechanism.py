"""Non-temporal async scheduling explanation, distinct from measured swimlanes."""


def mechanism_drawing():
    # Columns represent business dependencies, never elapsed-time positions.
    from ep128_swimlane_plot import _line, _rect, _text
    e = []
    width, height, left = 1600, 745, 170
    _rect(e, 0, 0, width, height, "FFFFFF")
    _text(e, 12, 8, width-24, 30, "异步调度机制：提交、依赖与等待（业务示意，横向不表示时间）", 16, bold=True)
    _text(e, 12, 42, width-24, 27, "n表示当前主模型验证及其候选生成；n+1表示使用这些候选的下一主轮。batch/RPC编号不自动等于HC轮号。", 10, "62758A")

    def box(x, y, w, title, lines, fill="EDF2F7", h=81):
        _rect(e, x, y, w, h, fill, stroke="A4B5C3", stroke_width=.8)
        _text(e, x+8, y+4, w-16, 24, title, 10, bold=True)
        for i, text in enumerate(lines):
            _text(e, x+8, y+29+21*i, w-16, 22, text, 9, "355169")

    labels = ((88, "EngineCore CPU"), (205, "Worker主线程 CPU"), (345, "NPU队列 / 事件"), (473, "输出线程 CPU"))
    for y, label in labels:
        _text(e, 10, y, left-17, 60, label, 11, bold=True)
    box(170, 88, 310, "调度 / 提交n轮RPC", ("execute_model(..., non_block=True)", "sample_tokens(..., non_block=True)"), "FFF1D5")
    box(540, 88, 400, "可继续提交 / 管理其他批次", ("非阻塞RPC返回future，不是设备结果完成", "回收哪一批，按其batch_id关联"), "FFF1D5")
    box(1110, 88, 450, "需要消费某批结果时才等待", ("future.result() → scheduler.update_from_output", "不把这次Host等待直接计作D2H时间"), "FFF1D5")
    _line(e, 480, 128, 540, 128, "AA4952", 1.3, arrow="end")
    _line(e, 940, 128, 1110, 128, "AA4952", 1.3, arrow="end")
    box(170, 205, 280, "execute_model(n)", ("准备输入 / metadata，提交主FULL图", "保存设备侧logits / hidden states"), "DFE8F1")
    box(480, 205, 430, "sample_tokens(n) 内含 propose(n)", ("提交采样/拒绝采样 → 草稿上下文/metadata", "提交草稿FULL图及候选生成；返回Tensor句柄"), "EAE2F3")
    box(940, 205, 620, "execute_model(n+1)：主线程接着做下一轮准备", ("请求前处理 → DP协调 → inputs → 块表/slot → metadata → 主FULL图", "真正需要Host可读结果或跨rank结果时，才在对应同步点阻塞"), "E9E1CD")
    for a, b in ((450,480),(910,940)):
        _line(e, a, 244, b, 244, "AA4952", 1.3, arrow="end")
    box(170, 345, 280, "主FULL graph(n)", ("主HC只是其中层链包络", "logits就绪后，设备采样才能消费"), "D7E8EE")
    box(480, 345, 430, "验证 → 草稿FULL图 → 候选生成", ("设备依赖由队列顺序 / stream-event满足", "CPU提交完成，不要求这里已执行完成"), "D5EBE6")
    box(940, 345, 620, "n+1输入整理 / 拷贝 / metadata任务 → 主FULL图", ("消费n轮采样/候选的任务必须等其设备结果就绪", "CPU可先准备不依赖结果的部分；具体流与依赖须查本轮trace"), "DDE9F0")
    for a, b in ((450,480),(910,940)):
        _line(e, a, 385, b, 385, "35829B", 1.2, "dash", arrow="end")
    for sx, tx in ((310,310),(695,695),(1250,1250)):
        _line(e, sx, 286, tx, 345, "35829B", 1.2, "dash", arrow="end")
    box(170, 473, 340, "AsyncOutput入队", ("Worker提交D2H / 完成事件（条件分支）", "返回输出对象，不等于Host已可读"), "ECEFF3")
    box(570, 473, 450, "输出线程取出 / materialize", ("等待输出完成事件，再CPU解析token", "与Worker主线程准备下一轮可并行"), "ECEFF3")
    box(1110, 473, 450, "发送结果 → EngineCore回收", ("响应绑定原batch_id", "没有导出这条线程的trace，不画实测耗时"), "ECEFF3")
    _line(e, 510, 514, 570, 514, "AA4952", 1.3, arrow="end")
    _line(e, 1020, 514, 1110, 514, "AA4952", 1.3, arrow="end")
    for a,b in (((925,244),(925,455)), ((925,455),(340,455))):
        _line(e, *a, *b, "AA4952", 1.1)
    _line(e, 340, 455, 340, 473, "AA4952", 1.1, arrow="end")
    _line(e, 1560, 514, 1578, 514, "AA4952", 1.1)
    _line(e, 1578, 514, 1578, 128, "AA4952", 1.1)
    _line(e, 1578, 128, 1560, 128, "AA4952", 1.1, arrow="end")
    _line(e, 695, 426, 795, 473, "35829B", 1.2, "dash", arrow="end")
    _text(e, 12, 585, width-24, 29, "必须分清三种等待：①CPU阻塞wait/synchronize；②设备stream/event依赖（CPU通常继续）；③EngineCore/输出线程等待另一批结果。", 11, bold=True)
    _text(e, 12, 619, width-24, 28, "实测图只为已导出的scope画真实起止；Worker业务条中的未知间隙≠CPU idle。父函数与内部子段不能相加，HC/U竖线不是函数调用分界。", 10, "536B81")
    _text(e, 12, 651, width-24, 28, "红线=业务先后/消息传递；蓝虚线=Host提交或设备数据依赖。这里均是机制关系，不是已证明的本轮flow，不标实测时长。", 10, "536B81")
    _text(e, 12, 683, width-24, 30, "结论：propose返回只说明Host工作及下发已走完；设备结果就绪由依赖保证。FULL graph保留。整轮收益须看主HC起点→下一主HC起点，不能只看U缩短。", 11, "264B67", bold=True)
    return {"width": width, "height": height, "elements": e, "title": "异步调度业务示意 · 非时间比例", "temporal": False,
            "description": "异步业务机制示意，横向不表示时间，非实测flow；不能当作本轮设备提交证据"}
