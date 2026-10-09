import docker
import re
from langchain_core.tools import tool
import time
docker_client = docker.from_env()


@tool
def execute_python_code_in_docker(code: str) -> str:
    """
    Executes Python code inside an isolated local Docker container and returns the output.
    Use this to run mathematical calculations, data formatting, or custom scripts safely.
    """
    cleaned_code = re.sub(r"^```python\s*|\s*```$", "", code.strip(), flags=re.MULTILINE)
    cmd = ["python", "-c", cleaned_code]
    
    container = None
    try:
        container = docker_client.containers.run(
            image="python:3.11-slim",
            command=cmd,
            detach=True,
            network_mode="none",       # Complete network isolation
            mem_limit="128m",          # Restrict RAM
            nano_cpus=1000000000,      # Cap CPU usage to 1 vCPU max
            read_only=True,            # Make filesystem read-only
            security_opt=["no-new-privileges:true"],
            user="nobody"              # Run as an unprivileged user
        )
        print(f"[DOCKER] Container created: {container.id}")
        print(f"[DOCKER] Status: {container.status}")
        
        result = container.wait(timeout=10)
        logs = container.logs(stdout=True, stderr=True).decode("utf-8")
        
        if result.get("StatusCode") != 0:
            return f"Runtime Error (Exit Code {result.get('StatusCode')}):\n{logs}"
        
        return logs if logs.strip() else "Code executed successfully with no output."

    except Exception as e:
        return f"Execution failed or timed out: {str(e)}"
    finally:
        if container:
            try:
                # time.sleep(30)
                container.remove(force=True)
            except Exception:
                pass
