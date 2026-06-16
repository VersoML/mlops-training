import os

from kfp import dsl


from orchestration.client import kfp_client


@dsl.component(base_image="python:3.12-slim")
def say_hello(name: str) -> str:
    msg = f"Hello, {name}!"
    print(msg)
    return msg


@dsl.component(base_image="python:3.12-slim")
def reverse(text: str) -> str:
    out = text[::-1]
    print(out)
    return out


@dsl.pipeline(name="hello-pipeline", description="A 2-step demo pipeline")
def hello_pipeline(name: str = "World") -> str:
    greet = say_hello(name=name)
    rev = reverse(text=greet.output)
    return rev.output


if __name__ == "__main__":
    run = kfp_client.create_run_from_pipeline_func(
        hello_pipeline,
        arguments={"name": "Kubeflow"},
        experiment_name="hello",
        namespace=os.environ["KUBEFLOW_NAMESPACE"],
    )
    endpoint = os.environ["KUBEFLOW_ENDPOINT"].rstrip("/")
    print(f"Run URL: {endpoint}/pipeline/#/runs/details/{run.run_id}")