//! DeepSeek 核心配置 —— 独立于根 crate 的 Config
//!
//! 由根 crate 的 `Config` 构造转换而来。

/// ds_core 所需的配置（从根 crate Config 的子集构造）
#[derive(Debug, Clone)]
pub struct DsCoreConfig {
    pub api_base: String,
    pub wasm_url: String,
    pub user_agent: String,
    pub client_version: String,
    pub client_platform: String,
    pub client_locale: String,
    /// X-Client-Bundle-Id 请求头（真实客户端固定为 com.deepseek.chat）
    pub client_bundle_id: String,
    /// X-Device-Id 请求头（设备级 UUID，空 = 按 api_base 确定性派生兜底）
    pub client_device_id: String,
    /// X-Device-Model 请求头（真实 Web 客户端发空串）
    pub client_device_model: String,
    /// X-Client-Timezone-Offset 请求头（分钟或秒，真实 Web 客户端 UTC+8 发 28800）
    pub client_timezone_offset: String,
    /// 登录 payload 的 os 字段（真实 Web 客户端为 "web"，App 为 "android"）
    pub client_os: String,
    /// 传输层拟态档位（`okhttp4_12` = 原生 App，`chrome136` = 桌面 Chrome）
    pub emulation: String,
    /// 是否在 completion 请求上回传 `x-hif-leim` 风控令牌
    ///
    /// 真实客户端会轮询 `hif-leim.deepseek.com` / `hif-dliq.deepseek.com` 取令牌
    /// 并在 SSE 请求上带回；关闭仅用于对照实验（正常使用务必保持开启）。
    pub hif_enabled: bool,
    /// 启动时是否用一条 completion 做健康检查（默认 `true`）
    ///
    /// 真实客户端启动只做「登录 → check_device → 拉会话列表」，**不发消息**；
    /// 置 `false` 可对齐该序列（账号可用性仍由登录 + 禁言早检保证），
    /// 用于风控对照实验。
    pub startup_health_check: bool,
    /// 会话策略：`per_request`（默认，每轮新建、结束时删除）或 `reuse`
    ///
    /// 真实客户端一个会话长期复用、几乎不删。`reuse` 会保留会话并在后续请求中复用，
    /// 空闲超过 `session_idle_secs` 才删除。**复用语义尚未用真实账号验证**，
    /// 仅用于对照实验；未知取值按 `per_request` 处理。
    pub session_policy: String,
    /// 复用模式下会话的空闲回收秒数（0 = 只在进程退出时删除）
    pub session_idle_secs: u64,
    pub proxy_url: Option<String>,
    pub model_types: Vec<String>,
    pub input_character_limits: Vec<u32>,
    /// 每账号每小时请求上限（0 = 不限制）
    pub hourly_request_quota: u64,
    /// 同一账号两次对话请求的最小间隔秒数（0 = 不限制）
    ///
    /// 真实用户不会以每秒一条的频率连续发消息；高频突发是上游行为风控最明显的
    /// 信号之一。该间隔在发出 completion 前生效（含会话创建/PoW 时间），并附带
    /// 20% 以内的抖动，避免形成精确的机械节拍。
    pub min_request_interval_secs: u64,
}

/// 单个账号配置
#[derive(Debug, Clone)]
pub struct AccountConfig {
    pub email: String,
    pub mobile: String,
    pub area_code: String,
    pub password: String,
    /// 浏览器设备指纹 ID。
    ///
    /// **实测为必填**：缺失会被登录风控直接拒绝（`RISK_DEVICE_DETECTED`，biz_code 11）。
    ///
    /// 建议**每个账号使用独立的 device_id**：设备级指纹被上游用于关联与画像，
    /// 同一指纹下挂多个账号、累计数百次请求后，账号会被禁言（`biz_code=5`）。
    pub device_id: String,
    /// 浏览器真实 `X-Device-Id` 请求头（可选）
    ///
    /// 它和数美 `device_id` 是**两个不同的值**：`device_id` 是登录 payload 里的
    /// 设备指纹，`X-Device-Id` 是浏览器页面内 XHR 请求头里的 UUID。
    /// 留空时按 `device_id` 派生一个稳定 UUID，但派生值并不是浏览器实际发出的值；
    /// 要最大程度对齐浏览器，应该从 F12 → Network → 任一 `/api/v0/*` 请求的
    /// 请求头里复制 `x-device-id` 填到这里。
    pub x_device_id: String,
    /// 浏览器导入的 Bearer token（非空 = token 模式）
    ///
    /// token 模式下**不再调用 `/users/login`**（也不调 `check_device`）：直接在
    /// 浏览器（`chat.deepseek.com`，F12 → Network → 任一 `/api/v0/*` 请求的
    /// `Authorization: Bearer …`）复制 token 使用。初始化只做一次
    /// `GET /users/current` 校验有效性并读取禁言状态。
    ///
    /// 注意：token 会过期，失效后账号被标记为 `invalid`，需要重新导入。
    pub token: String,
}
