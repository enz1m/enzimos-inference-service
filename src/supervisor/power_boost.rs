// SPDX-License-Identifier: MIT
use std::time::Duration;
use tokio::sync::oneshot;

const NAME: &str = "dev.enzui.PowerBoost1";
const PATH: &str = "/dev/enzui/PowerBoost1";
const DEADLINE: Duration = Duration::from_millis(500);
const REFRESH: Duration = Duration::from_millis(650);

pub(super) struct Boost {
    _work: oneshot::Sender<()>,
}

impl Boost {
    pub(super) fn while_processing() -> Self {
        let (work, done) = oneshot::channel();
        tokio::spawn(maintain(done));
        Self { _work: work }
    }
}

async fn maintain(mut done: oneshot::Receiver<()>) {
    let connected = tokio::select! {
        _ = &mut done => return,
        result = tokio::time::timeout(DEADLINE, zbus::Connection::system()) => result,
    };
    let Ok(Ok(bus)) = connected else {
        return;
    };
    let mut refresh = tokio::time::interval(REFRESH);
    loop {
        tokio::select! {
            biased;
            _ = &mut done => break,
            _ = refresh.tick() => {}
        }
        // Let an in-flight hold finish before releasing it on the same connection.
        let held = tokio::time::timeout(
            DEADLINE,
            bus.call_method(Some(NAME), PATH, Some(NAME), "Hold", &()),
        )
        .await;
        match held {
            Ok(Ok(reply))
                if reply.body().deserialize::<String>().is_ok_and(|outcome| {
                    matches!(outcome.as_str(), "boosted" | "extended" | "busy")
                }) => {}
            _ => break,
        }
    }
    let _ = tokio::time::timeout(
        DEADLINE,
        bus.call_method(Some(NAME), PATH, Some(NAME), "Release", &()),
    )
    .await;
}
