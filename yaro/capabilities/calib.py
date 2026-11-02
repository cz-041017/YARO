"""标定闭环：把"人肉调标定"变成"有判据、有解释、可回滚"的流程。

三种标定，优先级不同（见执行方案）：

- **A 内参与畸变**（P0）：棋盘格/ArUco → ``calibrateCameraExtended`` → 重投影误差 + 棋盘覆盖率
- **B 手眼标定**（P2）：``calibrateHandEye`` + 姿态多样性诊断
- **C 多相机/结构光/线激光**（P3）：联合优化，超出现阶段范围

硬约束（写在 CONTRACT.md 里，代码里也断言）：

1. **物理世界不可回滚** → 永远保留原图，标定结果只以"新产物 + 版本指针"出现；
2. **精度有物理上限** → RMS 收敛不到阈值时要能判定是"算法没调好"还是"硬件/采集不够"；
3. **失败要可解释** → 每条失败都要返回"下一步该做什么"，而不是一个 False。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..contracts import DegradedBackend, Evidence, ToolResult


@dataclass(frozen=True)
class CalibThresholds:
    """验收阈值。改这里必须同步改 contract/CONTRACT.md。"""

    max_rms_px: float = 0.3
    min_coverage: float = 0.6
    max_view_rms_px: float = 0.5
    min_views: int = 10


@dataclass
class CalibReport:
    ok: bool
    rms_px: float
    coverage: float
    views: int
    per_view_rms: list[float] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    thresholds: CalibThresholds = field(default_factory=CalibThresholds)

    def to_result(self) -> ToolResult:
        """转成工具结果：标量进 metrics（唯一可信分数来源），原因进证据。"""
        evidence = [
            Evidence(
                kind="table",
                caption="每视角重投影误差（px）",
                payload=[{"view": index, "rms_px": round(value, 4)} for index, value in enumerate(self.per_view_rms)],
            )
        ]
        summary = (
            f"标定{'通过' if self.ok else '未通过'}：RMS={self.rms_px:.3f}px（阈值 {self.thresholds.max_rms_px}）、"
            f"覆盖率={self.coverage:.0%}（阈值 {self.thresholds.min_coverage:.0%}）、有效视角 {self.views}"
        )
        if self.reasons:
            summary += "；原因：" + "；".join(self.reasons)
        return ToolResult(
            ok=self.ok,
            summary=summary,
            evidence=evidence,
            metrics={"rms_px": self.rms_px, "coverage": self.coverage, "views": float(self.views)},
            hint="；".join(self.next_actions) if self.next_actions else None,
        )


def assess_calibration(
    rms_px: float,
    coverage: float,
    views: int,
    per_view_rms: Sequence[float] | None = None,
    thresholds: CalibThresholds | None = None,
) -> CalibReport:
    """纯函数判据：不碰 IO、不碰 GPU，因此可以在 CI 里直接单测。

    ``coverage`` 定义：成功检出棋盘/标记并参与求解的视角占全部输入视角的比例。
    这比"棋盘是否铺满画面"更容易自动计算，也更能暴露"只在画面中央拍了 3 张"的偷懒。
    """
    thresholds = thresholds or CalibThresholds()
    per_view = list(per_view_rms or [])
    reasons: list[str] = []
    next_actions: list[str] = []

    if views < thresholds.min_views:
        reasons.append(f"有效视角太少（{views} < {thresholds.min_views}）")
        next_actions.append("补拍更多视角：棋盘需覆盖画面四角与边缘，姿态至少 10 组，含倾斜 30° 以上")

    if rms_px > thresholds.max_rms_px:
        reasons.append(f"总体重投影误差超标（{rms_px:.3f}px > {thresholds.max_rms_px}px）")
        if coverage < thresholds.min_coverage:
            next_actions.append("先提高覆盖率（补拍边缘与远场），再重新求解；覆盖不足时 RMS 会被中央视角拉低而失真")
        else:
            next_actions.append(
                "检查镜头畸变模型（尝试 rational/薄棱镜）、剔除每视角误差最大的 2–3 张、"
                "确认棋盘格实际尺寸与标称一致"
            )

    if coverage < thresholds.min_coverage:
        reasons.append(f"棋盘覆盖率不足（{coverage:.0%} < {thresholds.min_coverage:.0%}）")
        next_actions.append("让棋盘出现在画面边缘/四角，不要只拍中心区域")

    outliers = [index for index, value in enumerate(per_view) if value > thresholds.max_view_rms_px]
    if len(outliers) >= max(2, len(per_view) // 5) and per_view:
        reasons.append(f"{len(outliers)} 个视角误差超过 {thresholds.max_view_rms_px}px（视角 {outliers[:8]}）")
        next_actions.append("剔除这些视角后重解；若反复出现，检查该区域的运动模糊与过曝")

    return CalibReport(
        ok=not reasons,
        rms_px=float(rms_px),
        coverage=float(coverage),
        views=int(views),
        per_view_rms=per_view,
        reasons=reasons,
        next_actions=next_actions,
        thresholds=thresholds,
    )


def checkerboard_object_points(
    cols: int,
    rows: int,
    square_size: float,
) -> list[list[tuple[float, float, float]]]:
    """棋盘格的三维物点（纯标准库实现，方便在无 cv2 环境里单测）。

    ``cols``/``rows`` 是**内角点**数量，不是方格数。
    """
    if cols < 2 or rows < 2:
        raise ValueError("内角点必须至少 2×2")
    if square_size <= 0:
        raise ValueError("方格边长必须为正（单位与后续平移量一致，建议米）")
    points: list[tuple[float, float, float]] = []
    for row in range(rows):
        for col in range(cols):
            points.append((col * square_size, row * square_size, 0.0))
    return [points]


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - 取决于安装的 extras
        raise DegradedBackend(
            "缺少 opencv：pip install 'yaro[cv]'。标定求解需要 OpenCV，"
            "但判据 assess_calibration 是纯 Python 的，可以先行接入 CI。"
        ) from exc
    return cv2


def calibrate_intrinsics(
    image_paths: Sequence[str],
    *,
    pattern: tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    thresholds: CalibThresholds | None = None,
) -> CalibReport:
    """从棋盘格图像求内参与畸变（相机矩阵 + 5 阶畸变）。

    返回 ``CalibReport``；调用方负责把 K/dist 作为产物写入 CAS（不允许原地覆盖旧标定）。
    """
    cv2 = _import_cv2()
    import numpy as np  # noqa: PLC0415 - 与 cv2 同属可选依赖

    cols, rows = pattern
    object_points = [
        np.array(checkerboard_object_points(cols, rows, square_size)[0], dtype=np.float32)
    ]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)

    object_views: list[Any] = []
    image_views: list[Any] = []
    per_view: list[float] = []
    image_size: tuple[int, int] | None = None
    attempted = len(image_paths)

    for path in image_paths:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        image_size = image_size or (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCornersSB(gray, (cols, rows))
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_views.append(object_points[0])
        image_views.append(refined)
        per_view.append(0.0)  # 先占位，求解后回填

    if len(object_views) < 4 or image_size is None:
        return CalibReport(
            ok=False,
            rms_px=float("inf"),
            coverage=(len(object_views) / attempted) if attempted else 0.0,
            views=len(object_views),
            reasons=["可用于求解的棋盘视角不足 4 个"],
            next_actions=["确认棋盘内角点数量与 pattern 一致、图像不过曝、棋盘平整且占画面 1/3 以上"],
            thresholds=thresholds or CalibThresholds(),
        )

    result = cv2.calibrateCameraExtended(
        object_views,
        image_views,
        image_size,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )
    # OpenCV 返回 8 项：retval, K, dist, rvecs, tvecs, stdIntrinsics, stdExtrinsics, perViewErrors
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs, _std_intr, _std_extr, per_view_errors = result
    errors = [float(value) for value in np.asarray(per_view_errors).ravel()]
    coverage = len(object_views) / attempted if attempted else 0.0

    report = assess_calibration(
        rms_px=float(rms),
        coverage=coverage,
        views=len(object_views),
        per_view_rms=errors,
        thresholds=thresholds,
    )
    report.per_view_rms = errors
    return report


def calibrate_intrinsics_matrices(
    image_paths: Sequence[str],
    *,
    pattern: tuple[int, int] = (9, 6),
    square_size: float = 0.025,
) -> tuple[CalibReport, Any, Any]:
    """同上，但额外返回 ``(K, dist)``。

    单独实现是为了让"判据"与"矩阵"解耦：多数调用方只关心过没过。
    """
    cv2 = _import_cv2()
    import numpy as np  # noqa: PLC0415

    cols, rows = pattern
    template = np.array(checkerboard_object_points(cols, rows, square_size)[0], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
    object_views, image_views, image_size = [], [], None
    attempted = len(image_paths)

    for path in image_paths:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        image_size = image_size or (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCornersSB(gray, (cols, rows))
        if not found:
            continue
        object_views.append(template)
        image_views.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))

    if len(object_views) < 4 or image_size is None:
        raise DegradedBackend(f"可用棋盘视角不足（{len(object_views)} 个），无法求解内参")

    rms, k, dist, _rvecs, _tvecs, _std_intr, _std_extr, per_view_errors = cv2.calibrateCameraExtended(
        object_views, image_views, image_size, None, None, flags=cv2.CALIB_RATIONAL_MODEL
    )
    report = assess_calibration(
        rms_px=float(rms),
        coverage=len(object_views) / attempted if attempted else 0.0,
        views=len(object_views),
        per_view_rms=[float(value) for value in np.asarray(per_view_errors).ravel()],
    )
    return report, k, dist
