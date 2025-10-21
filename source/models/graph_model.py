from typing import Any, Dict, List, Tuple, Union, Optional
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.cli import instantiate_class
from torch import nn
import torch
        
    
class GraphModel(nn.Module):
    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        graph_cfg: DictConfig,
    ):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.graph_cfg = graph_cfg
        
        self.nodes = self._create_modules(graph_cfg)
        self.execution_order = self._topological_sort(graph_cfg)
        pass


    def _create_modules(self, graph_cfg: DictConfig) -> nn.ModuleDict:
        nodes = nn.ModuleDict()
        
        for name, cfg in graph_cfg.items():
            if 'module' not in cfg:
                raise ValueError(f"Node '{name}' must have a 'module' definition.")
            nodes[name] = instantiate_class(args=(), init=cfg['module'])
            
        return nodes


    def _topological_sort(self, graph_cfg: DictConfig) -> List[str]:
        all_node_names = set(self.inputs) | set(graph_cfg.keys())
        in_degree = {name: 0 for name in all_node_names}
        graph_connections = {name: [] for name in all_node_names}

        for name, cfg in graph_cfg.items():
            source_paths = []
            
            # 'inputs' 필드 처리: 단일 실행 모드
            if 'inputs' in cfg:
                node_source = cfg['inputs']
                if isinstance(node_source, (DictConfig, dict)):
                    source_paths = list(node_source.values())
                elif isinstance(node_source, (ListConfig, list)):
                    source_paths = node_source
                else: # String
                    source_paths = [node_source]
            
            # 'inputs_for' 필드 처리: 반복 실행 모드
            elif 'inputs_for' in cfg:
                for input_structure in cfg['inputs_for'].values():
                    if isinstance(input_structure, (DictConfig, dict)):
                        source_paths.extend(input_structure.values())
                    elif isinstance(input_structure, (ListConfig, list)):
                        source_paths.extend(input_structure)
                    else: # String
                        source_paths.append(input_structure)

            for src in source_paths:
                src_base = src.split('.')[0]
                
                if src_base not in all_node_names:
                    raise KeyError(f"Source base '{src_base}' for node '{name}' is not defined.")
                
                graph_connections[src_base].append(name)
                in_degree[name] += 1
        
        queue = [name for name in self.inputs]
        sorted_order = []
        while queue:
            u = queue.pop(0)
            sorted_order.append(u)
            if u in graph_connections:
                for v in graph_connections[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
        
        executable_nodes = [name for name in sorted_order if name not in self.inputs]
        if len(executable_nodes) != len(graph_cfg):
             raise ValueError("Graph has a cycle or disconnected nodes.")
        return executable_nodes


    def _recursive_find(self, current_obj: Any, remaining_parts: List[str]) -> Optional[Any]:
        if not remaining_parts:
            return current_obj
        
        next_key = remaining_parts[0]

        if isinstance(current_obj, (DictConfig, ListConfig, dict)):
            next_obj = current_obj.get(next_key)
        else:
            next_obj = getattr(current_obj, next_key, None)

        if next_obj is None:
            return None
        
        return self._recursive_find(next_obj, remaining_parts[1:])


    def _get_output_from_pool(self, pool: Dict[str, Any], source_name: str) -> Any:
        # '.'을 사용한 재귀적 접근을 처리합니다.
        parts = source_name.split('.')
        initial_source = pool.get(parts[0])
        
        if initial_source is None:
            raise KeyError(f"Source base '{parts[0]}' from path '{source_name}' not found in node outputs.")

        if len(parts) == 1:
            return initial_source
        
        resolved = self._recursive_find(initial_source, parts[1:])

        if resolved is None:
            raise KeyError(f"Failed to resolve path '{source_name}' in node outputs. Check the structure of '{parts[0]}' node's output.")

        return resolved


    def _resolve_input_structure(self, input_structure: Any, node_outputs: Dict[str, Any]) -> Optional[Union[Tuple[Any, ...], Dict[str, Any]]]:
        if input_structure is None:
            return None

        if isinstance(input_structure, (DictConfig, dict)):
            # Dict source: Module(**kwargs) 호출을 위해 Dict 형태로 반환
            return {
                arg_name: self._get_output_from_pool(node_outputs, source_name)
                for arg_name, source_name in input_structure.items()
            }
        
        elif isinstance(input_structure, (ListConfig, list)):
            # List source: Module(*args) 호출을 위해 Tuple 형태로 반환
            resolved_inputs = [self._get_output_from_pool(node_outputs, src) for src in input_structure]
            return tuple(resolved_inputs)
            
        else: 
            # Single path source (String): Module(*args) 호출을 위해 Tuple 형태로 반환
            resolved_input = self._get_output_from_pool(node_outputs, input_structure)
            return resolved_input if isinstance(resolved_input, tuple) else (resolved_input,)


    def forward(self, batch: Dict[str, torch.Tensor], **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        node_outputs = batch

        for name in self.execution_order:
            cfg = self.graph_cfg[name]
            module = self.nodes[name]
            
            module_input = None
            
            # 1. 'inputs_for' 처리 (반복 실행)
            if 'inputs_for' in cfg:
                mapped_outputs = {}
                for suffix, input_structure in cfg['inputs_for'].items():
                    # 'inputs_for'의 각 항목(e.g., 't', 'v')마다 입력 구조를 해석하여 모듈에 전달합니다.
                    arg_tuple_or_dict = self._resolve_input_structure(input_structure, node_outputs)

                    if arg_tuple_or_dict is None:
                        mapped_outputs[suffix] = module()
                    elif isinstance(arg_tuple_or_dict, tuple):
                        mapped_outputs[suffix] = module(*arg_tuple_or_dict)
                    elif isinstance(arg_tuple_or_dict, dict):
                        mapped_outputs[suffix] = module(**arg_tuple_or_dict)
                    else:
                        mapped_outputs[suffix] = module(arg_tuple_or_dict)

                node_outputs[name] = mapped_outputs
                continue

            # 2. 'inputs' 처리 (단일 실행)
            module_input = self._resolve_input_structure(cfg.get('inputs'), node_outputs)

            # 3. Module 실행
            if module_input is None:
                node_outputs[name] = module()
            elif isinstance(module_input, tuple):
                node_outputs[name] = module(*module_input)
            elif isinstance(module_input, dict):
                node_outputs[name] = module(**module_input)
            else:
                node_outputs[name] = module(module_input) 

        final_outputs = {name: self._get_output_from_pool(node_outputs, name) for name in self.outputs}
        return final_outputs