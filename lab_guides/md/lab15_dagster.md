Intro to ops and jobs

Dagster's asset functionality sits on top of a general orchestration engine that can be used for tasks other than creating and maintaining assets.


#### Prerequisites

To complete this lab, you'll need to install Dagster, the Dagster webserver/UI, and the requests library:

`pip install dagster dagster-webserver`

This installs a few packages:

- `dagster`. The core programming model and abstraction stack; stateless, single-node, single-process and multi-process execution engines; and a CLI tool for driving those engines. Refer to the Dagster installation guide for more info, including how to ensure your environment is set up correctly.
- `dagster-webserver`: The server for Dagster's browser UI for developing and operating Dagster jobs, including a DAG browser, a type-aware config editor, and a live execution interface.