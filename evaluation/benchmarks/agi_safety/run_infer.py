import asyncio
import base64
import copy
import io
import json
import os
import tarfile
from typing import Any
import random 
import pandas as pd
from PIL import Image

from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    codeact_user_response,
    compatibility_for_eval_history_pairs,
    get_default_sandbox_config_for_eval,
    get_metrics,
    get_openhands_config_for_eval,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    OpenHandsConfig,
    get_evaluation_parser,
    get_llm_config_arg,
    load_from_toml,
)
from openhands.core.config.utils import (
    get_agent_config_arg,
    get_llms_for_routing_config,
    get_model_routing_config_arg,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import MessageAction
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}


def image_to_base64_url(image_path: str) -> str:
    """Convert an image file to a base64 encoded data URL.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded data URL string
    """
    try:
        image = Image.open(image_path)
        # Convert RGBA/LA to RGB
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
        
        buffered = io.BytesIO()
        # Detect format from extension
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            image.save(buffered, format='JPEG')
            mime_type = 'image/jpeg'
        elif ext == '.png':
            image.save(buffered, format='PNG')
            mime_type = 'image/png'
        else:
            # Default to PNG for other formats
            image.save(buffered, format='PNG')
            mime_type = 'image/png'
        
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f'data:{mime_type};base64,{image_base64}'
    except Exception as e:
        logger.error(f'Failed to convert image {image_path} to base64: {e}')
        return ''



def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = 'nikolaik/python-nodejs:python3.12-nodejs22'
    # sandbox_config.base_container_image = 'python:3.12-bookworm'

    # Set volume mount as comma-delimited string: host_path:container_path:mode
    # Multiple mounts can be separated by commas
    sandbox_config.volumes = '/home/kzhou35/agi_safety/gdpval/reference_files:/workspace/reference_files:ro'
    
    config = get_openhands_config_for_eval(
        metadata=metadata,
        sandbox_config=sandbox_config,
        runtime='docker',
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['task_id']
        )
    )
    model_routing_config = get_model_routing_config_arg()
    model_routing_config.llms_for_routing = get_llms_for_routing_config()

    if metadata.agent_config:
        metadata.agent_config.model_routing = model_routing_config
        config.set_agent_config(metadata.agent_config, metadata.agent_class)
    else:
        logger.info('Agent config not provided, using default settings')
        agent_config = config.get_agent_config(metadata.agent_class)
        agent_config.enable_prompt_extensions = False
        agent_config.model_routing = model_routing_config

    config_copy = copy.deepcopy(config)
    load_from_toml(config_copy)
    config.search_api_key = config_copy.search_api_key
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
):
    """Initialize the runtime for the agent."""
    logger.info(f'{"-" * 50} BEGIN Runtime Initialization Fn {"-" * 50}')
    
    from openhands.events.action import CmdRunAction
    
    # Create workspace
    action = CmdRunAction(command='mkdir -p /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0
    
    # Change to workspace directory
    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0
    
    # Verify reference files are mounted
    action = CmdRunAction(command='ls -la /workspace/reference_files 2>&1 | head -20')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(f'Reference files directory contents: {obs.content}')
    
    # Install and verify packages are installed
    action = CmdRunAction(command='pip install python-docx reportlab python-pptx openpyxl pandas matplotlib pypdf cadquery svglib')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)

    action = CmdRunAction(command='python -c "import docx, reportlab, pptx, openpyxl, pandas, matplotlib, pypdf, cadquery, svglib; print(\'All packages installed successfully\')"')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(f'Package verification: {obs.content}')
    
    logger.info(f'{"-" * 50} END Runtime Initialization Fn {"-" * 50}')


def copy_container_outputs(
    runtime: Runtime,
    task_id: str,
    eval_output_dir: str,
) -> None:
    """Copy output files from container to host machine.
    
    Args:
        runtime: The runtime instance with access to the container
        task_id: The task ID for organizing outputs
        eval_output_dir: The evaluation output directory on the host
    """
    from openhands.events.action import CmdRunAction
    
    logger.info(f'Copying container outputs for task {task_id}...')
    
    # Create output directory on host
    host_output_dir = os.path.join(eval_output_dir, 'agent_outputs', task_id)
    os.makedirs(host_output_dir, exist_ok=True)
    
    # List of container paths to copy (you can modify this based on your needs)
    container_paths = [
        '/workspace/output',  # Common output directory
        '/workspace/*.pdf',   # PDF files
        '/workspace/*.docx',  # Word documents
        '/workspace/*.pptx',  # PowerPoint files
        '/workspace/*.xlsx',  # Excel files
        '/workspace/*.png',   # Images
        '/workspace/*.jpg',
        '/workspace/*.txt',   # Text files
        '/workspace/*.json',  # JSON files
    ]
    
    # Check if container is accessible
    if not hasattr(runtime, 'container') or runtime.container is None:
        logger.warning('Container not accessible, skipping output copy')
        return
    
    for container_path in container_paths:
        try:
            # Check if path exists in container
            action = CmdRunAction(command=f'ls -la {container_path} 2>/dev/null')
            obs = runtime.run_action(action)
            
            if obs.exit_code != 0:
                # Path doesn't exist, skip
                continue
            
            logger.info(f'Found files matching: {container_path}')
            
            # For glob patterns, we need to handle them differently
            if '*' in container_path:
                # Get list of matching files
                action = CmdRunAction(command=f'find {os.path.dirname(container_path)} -maxdepth 1 -name "{os.path.basename(container_path)}" -type f')
                obs = runtime.run_action(action)
                
                if obs.exit_code == 0 and obs.content.strip():
                    matching_files = obs.content.strip().split('\n')
                    for file_path in matching_files:
                        file_path = file_path.strip()
                        if file_path:
                            _copy_single_path(runtime.container, file_path, host_output_dir)
            else:
                # Copy directory or single file
                _copy_single_path(runtime.container, container_path, host_output_dir)
                
        except Exception as e:
            logger.warning(f'Could not copy {container_path}: {e}')
    
    logger.info(f'Finished copying outputs to {host_output_dir}')


def _copy_single_path(container, container_path: str, host_dest_dir: str) -> None:
    """Copy a single file or directory from container to host.
    
    Args:
        container: Docker container object
        container_path: Path in the container to copy
        host_dest_dir: Destination directory on host
    """
    import tarfile
    import io
    
    try:
        # Get tar archive from container
        bits, stat = container.get_archive(container_path)
        
        # Extract tar to host directory
        tar_stream = io.BytesIO()
        for chunk in bits:
            tar_stream.write(chunk)
        tar_stream.seek(0)
        
        with tarfile.open(fileobj=tar_stream, mode='r') as tar:
            tar.extractall(path=host_dest_dir)
        
        logger.info(f'Copied {container_path} to {host_dest_dir}')
        
    except Exception as e:
        logger.debug(f'Could not copy {container_path}: {e}')


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(instance, metadata)

    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.get("task_id", str(random.randint(100000, 999999))), log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.get("task_id", str(random.randint(100000, 999999)))}.')

    instruction = ""
    
    # append messages to the instruction
    messages = instance.get('messages', [])
    if messages and len(messages) > 0:
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                sender = msg.get('sender', 'Unknown')
                content = msg.get('content', '')
                instruction += f"\nMessage from {sender}:\n{content}\n"
    
    # Prepare image URLs list for image inputs
    image_urls = []
    
    # Add information about reference files if available
    reference_files = instance.get('reference_files', [])
    task_id = instance['task_id']
    
    # Define image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    if reference_files and len(reference_files) > 0:
        # Separate image files from other files
        image_files = []
        other_files = []
        
        for ref_file in reference_files:
            file_ext = os.path.splitext(ref_file)[1].lower()
            if file_ext in image_extensions:
                image_files.append(ref_file)
            else:
                other_files.append(ref_file)
        
        # Process image files - convert to base64 URLs for the LLM
        if image_files:
            instruction += "\n\nIMPORTANT: The following images are provided as inputs for this task:\n"
            for idx, img_file in enumerate(image_files, 1):
                # Construct the full path to the image file
                # The reference files are mounted in the workspace as reference_files/{task_id}/{filename}
                img_filename = os.path.basename(img_file)
                
                # For local loading, construct path from the gdpval dataset
                gdpval_base = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'gdpval')
                local_img_path = os.path.join(gdpval_base, ref_file)
                
                if os.path.exists(local_img_path):
                    # Convert image to base64 URL
                    img_url = image_to_base64_url(local_img_path)
                    if img_url:
                        image_urls.append(img_url)
                        instruction += f"- Image {idx}: {img_filename}\n"
                        logger.info(f'Added image {img_filename} as input')
                else:
                    logger.warning(f'Image file not found: {local_img_path}')
            
            instruction += "\nThese images have been provided as visual inputs. Please examine them carefully to complete the task.\n"
        
        # List all reference files (both images and others) available in the workspace
        instruction += "\n\nIMPORTANT: Reference files for this task are available in the following location:\n"
        instruction += f"reference_files/{task_id}/\n\n"
        instruction += "The following reference files are available for this task:\n"
        for ref_file in reference_files:
            # Extract just the filename from the path
            file_path = ref_file
            instruction += f"- {file_path}\n"
        instruction += "\nPlease use these reference files as needed to complete the task.\n"
    
    # Add information about available Python packages
    instruction += "\n"
    instruction += "Available Python packages for file reading, creation and editing:\n"
    instruction += "- python-docx: for creating and editing Word documents (.docx)\n"
    instruction += "- python-pptx: for creating and editing PowerPoint presentations (.pptx)\n"
    instruction += "- openpyxl: for creating and editing Excel spreadsheets (.xlsx)\n"
    instruction += "- pandas: for data manipulation and analysis\n"
    instruction += "- pypdf: for extracting text from PDF files\n"
    instruction += "- svglib: for reading and converting SVG files\n"
    instruction += "- reportlab: for creating PDF documents\n"
    instruction += "- cadquery: for creating and manipulating 3D CAD models\n"

    logger.info(f'Instruction: {instruction}')

    # Create runtime and run controller
    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)
    
    try:
        initialize_runtime(runtime, instance)
        
        state: State = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=MessageAction(
                    content=instruction,
                    image_urls=image_urls if image_urls else None
                ),
                runtime=runtime,
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
                    metadata.agent_class
                ),
            )
        )
        
        if state is None:
            raise ValueError('State should not be None.')
        
        # Extract all agent outputs from history
        agent_outputs = []
        for event in state.history:
            if event.source == 'agent':
                if hasattr(event, 'content') and event.content:
                    agent_outputs.append(str(event.content))
                elif hasattr(event, 'thought') and event.thought:
                    agent_outputs.append(str(event.thought))
        
        model_output = '\n\n'.join(agent_outputs)
        
        # Get metrics
        metrics = get_metrics(state)
        
        # Prepare history for compatibility
        histories = compatibility_for_eval_history_pairs(state.history)
        
        # Prepare test result
        test_result = {
            'task_id': instance.get('task_id', str(random.randint(100000, 999999))),
            'category': instance.get('category', ''),
            'prompt': instance.get('prompt', ''),
            'agent_output': model_output.strip(),
            'risk_category': instance.get('risk_category', ''),
            'malicious_motivation': instance.get('malicious_motivation', ''),
            'risk_outcome': instance.get('risk_outcome', ''),
            'unsafe_criteria': instance.get('unsafe_criteria', ''),
            'safe_criteria': instance.get('safe_criteria', ''),
        }
        
        # Prepare output
        output = EvalOutput(
            instance_id=instance['instance_id'],
            instance=instance.to_dict(),
            instruction=instruction,
            metadata=metadata,
            history=histories,
            metrics=metrics,
            error=state.last_error if state and state.last_error else None,
            test_result=test_result,
        )
        
        return output
        
    finally:
        # Copy output files from container to host before closing
        try:
            copy_container_outputs(runtime, instance['task_id'], metadata.eval_output_dir)
        except Exception as e:
            logger.error(f'Failed to copy container outputs: {e}')
        
        runtime.close()


def load_safety_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the safety test dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Flatten the nested structure
    instances = []
    for category, tasks in data.items():
        for task in tasks:
            task['category'] = category
            instances.append(task)
    
    for i, instance in enumerate(instances):
        instance['instance_id'] = f'safety_test_{i+1}'
    
    return pd.DataFrame(instances)


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='/home/kzhou35/agi_safety/safety_tests.json',
        help='Path to the safety tests JSON file',
    )
    args, _ = parser.parse_known_args()
    
    # Load dataset
    dataset = load_safety_dataset(args.dataset_path)
    logger.info(f'Loaded {len(dataset)} safety test instances')
    
    logger.info(f'Loaded llm config: {args.llm_config}')
    llm_config = get_llm_config_arg(args.llm_config) if args.llm_config else None
    if llm_config:
        llm_config.modify_params = False
        # For Bedrock models, disable top_p since it conflicts with temperature
        if 'bedrock' in llm_config.model.lower():
            llm_config.top_p = None
    
    agent_config = get_agent_config_arg(args.agent_config) if args.agent_config else None
    
    metadata = make_metadata(
        llm_config=llm_config,
        dataset_name='agi_safety',
        agent_class=args.agent_cls,
        max_iterations=args.max_iterations,
        eval_note=args.eval_note,
        eval_output_dir=args.eval_output_dir,
        data_split=None,
        agent_config=agent_config,
    )
    
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(dataset, output_file, args.eval_n_limit)
    
    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
    )