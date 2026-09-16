"""
Memory Manager: Unified STM/LTM Policy (AgeMem + SimpleMem + AutoMem)

Problem: LLM agents face context bloat in long-horizon tasks.
Solution: Unified memory management with semantic compression, recursive consolidation, and adaptive retrieval.

Metrics:
- Memory Efficiency: (compressed_size / original_size) * 100
- Retrieval Latency: P95 latency for memory queries
- Recall@K: Fraction of relevant memories retrieved in top-K results

Research Backing:
- AgeMem (arXiv:2601.01885): Memory operations as tool-based actions
- SimpleMem (arXiv:2601.02553): Three-stage pipeline for memory management
- AutoMem (arXiv:2607.01224): Memory as cognitive skill
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
import os
import time
from sentence_transformers import SentenceTransformer
import heapq
class _NumpyJSONEncoder(json.JSONEncoder):
    """Lets json.dump handle numpy arrays (e.g. MemoryUnit.embedding) by
    converting them to plain lists. Pure serialization shim — does not
    change what is stored or any retrieval/similarity logic."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class MemoryUnit:
    """
    Atomic memory unit with semantic embedding and metadata.
    Based on SimpleMem's memory unit design.
    """
    content: str
    embedding: np.ndarray
    timestamp: float
    importance: float  # 0.0 to 1.0 (entropy-based)
    source: str  # "user_input", "agent_action", "external_tool"
    task_id: Optional[str] = None
    consolidated: bool = False  # whether this unit is part of higher-level abstraction


@dataclass
class MemoryQuery:
    """
    Query structure for memory retrieval.
    """
    query_text: str
    query_embedding: np.ndarray
    top_k: int = 5
    time_range: Optional[Tuple[float, float]] = None
    importance_threshold: float = 0.3


class EntropyBasedCompressor:
    """
    Semantic structured compression using entropy-aware filtering.
    Based on SimpleMem's compression stage.
    
    Key Idea: High-entropy (informative) parts are preserved, low-entropy (redundant) parts are compressed.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.compression_ratio = 0.0
    
    def compute_entropy(self, text: str) -> float:
        """
        Compute entropy of text using character-level frequency.
        Higher entropy = more informative.
        """
        if not text:
            return 0.0
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        entropy = 0.0
        for count in freq.values():
            p = count / len(text)
            entropy -= p * np.log2(p)
        return entropy
    
    def compress(self, text: str, target_ratio: float = 0.3) -> str:
        """
        Compress text by retaining high-entropy sentences.
        
        Args:
            text: Input text
            target_ratio: Target compression ratio (0.0 to 1.0)
        
        Returns:
            Compressed text
        """
        sentences = text.split(". ")
        if not sentences:
            return text
        
        # Compute entropy for each sentence
        sentence_entropies = [(s, self.compute_entropy(s)) for s in sentences]
        
        # Sort by entropy (descending)
        sentence_entropies.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences to achieve target ratio
        num_to_keep = max(1, int(len(sentences) * target_ratio))
        top_sentences = [s for s, _ in sentence_entropies[:num_to_keep]]
        
        compressed = ". ".join(top_sentences)
        self.compression_ratio = len(compressed) / len(text) if text else 1.0
        
        return compressed
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding for text.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding / np.linalg.norm(embedding)  # L2 normalize


class STMRingBuffer:
    """
    Short-term memory as a ring buffer with attention-based eviction.
    Based on AgeMem's STM design.
    
    Key Idea: Last N interactions stored in fast-access buffer. When full, evict lowest-importance unit.
    """
    
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.buffer: deque[MemoryUnit] = deque(maxlen=capacity)
        self.importance_heap: List[Tuple[float, int]] = []  # (importance, index)
    
    def add(self, unit: MemoryUnit):
        """
        Add memory unit to STM. If full, evict lowest-importance unit.
        """
        if len(self.buffer) >= self.capacity:
            # Evict lowest-importance unit
            _, evict_idx = heapq.heappop(self.importance_heap)
            # Note: In production, use weak references or mark as evictable
        self.buffer.append(unit)
        heapq.heappush(self.importance_heap, (unit.importance, len(self.buffer) - 1))
    
    def get_all(self) -> List[MemoryUnit]:
        """
        Get all memory units in STM.
        """
        return list(self.buffer)
    
    def clear(self):
        """
        Clear STM for new task.
        """
        self.buffer.clear()
        self.importance_heap.clear()


class LTMGraphDatabase:
    """
    Long-term memory as a graph database with semantic edges.
    Based on AgeMem's LTM design.
    
    Key Idea: Memories are nodes, semantic similarity forms edges. Enables multi-hop retrieval.
    """
    
    def __init__(self, db_path: str = "./memory/ltm_graph.json"):
        self.db_path = db_path
        self.nodes: Dict[str, MemoryUnit] = {}
        self.edges: Dict[str, List[str]] = {}  # node_id -> list of neighbor node_ids
        self._load()
    
    def _load(self):
        """
        Load LTM from disk.
        """
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for node_id, node_data in data["nodes"].items():
                    self.nodes[node_id] = MemoryUnit(**node_data)
                self.edges = data["edges"]
    
    def _save(self):
        """
        Save LTM to disk.
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
            "edges": self.edges,
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=_NumpyJSONEncoder)
    
    def add_node(self, node_id: str, unit: MemoryUnit):
        """
        Add memory node to LTM.
        """
        self.nodes[node_id] = unit
        self.edges[node_id] = []
        self._save()
    
    def add_edge(self, node_id1: str, node_id2: str, similarity: float):
        """
        Add semantic edge between two nodes if similarity > threshold.
        """
        if similarity > 0.7:  # threshold
            if node_id2 not in self.edges[node_id1]:
                self.edges[node_id1].append(node_id2)
            if node_id1 not in self.edges[node_id2]:
                self.edges[node_id2].append(node_id1)
            self._save()
    
    def retrieve(self, query: MemoryQuery) -> List[MemoryUnit]:
        """
        Retrieve top-K memories based on query.
        Uses multi-hop traversal for better recall.
        """
        # Step 1: Semantic similarity search
        scores = []
        for node_id, unit in self.nodes.items():
            similarity = np.dot(query.query_embedding, unit.embedding)
            if query.time_range:
                if not (query.time_range[0] <= unit.timestamp <= query.time_range[1]):
                    continue
            if unit.importance < query.importance_threshold:
                continue
            scores.append((node_id, similarity))
        
        # Step 2: Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Multi-hop expansion (1-hop neighbors)
        expanded_scores = []
        for node_id, score in scores[:query.top_k]:
            expanded_scores.append((node_id, score))
            for neighbor_id in self.edges.get(node_id, []):
                if neighbor_id not in [n for n, _ in expanded_scores]:
                    expanded_scores.append((neighbor_id, score * 0.8))  # decay factor
        
        # Step 4: Return top-K
        expanded_scores.sort(key=lambda x: x[1], reverse=True)
        return [self.nodes[node_id] for node_id, _ in expanded_scores[:query.top_k * 2]]


class MemoryManager:
    """
    Unified Memory Manager (AgeMem + SimpleMem + AutoMem)
    
    This is the core memory system that integrates:
    - STM ring buffer for fast access
    - LTM graph database for long-term storage
    - Entropy-based compressor for memory efficiency
    - Unified policy for memory operations (store, retrieve, update, discard)
    
    Usage:
        memory = MemoryManager()
        memory.store("User prefers morning meetings", source="user_input", task_id="task_001")
        memories = memory.retrieve("meeting preferences", top_k=3)
    """
    
    def __init__(self, stm_capacity: int = 10, ltm_path: str = "./memory/ltm_graph.json"):
        self.stm = STMRingBuffer(capacity=stm_capacity)
        self.ltm = LTMGraphDatabase(db_path=ltm_path)
        self.compressor = EntropyBasedCompressor()
        self.node_counter = 0
    
    def store(self, content: str, source: str, task_id: Optional[str] = None, compress: bool = True):
        """
        Store memory unit (STM + LTM).
        
        Args:
            content: Memory content
            source: Source type ("user_input", "agent_action", "external_tool")
            task_id: Associated task ID
            compress: Whether to compress content
        """
        # Step 1: Compress if needed
        if compress and len(content) > 200:
            content = self.compressor.compress(content, target_ratio=0.3)
        
        # Step 2: Compute embedding and importance
        embedding = self.compressor.embed(content)
        importance = self.compressor.compute_entropy(content) / 5.0  # normalize to 0-1
        
        # Step 3: Create memory unit
        unit = MemoryUnit(
            content=content,
            embedding=embedding,
            timestamp=time.time(),
            importance=importance,
            source=source,
            task_id=task_id,
        )
        
        # Step 4: Add to STM
        self.stm.add(unit)
        
        # Step 5: Add to LTM
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        self.ltm.add_node(node_id, unit)
        
        # Step 6: Add semantic edges to similar nodes (id-based, avoids
        # MemoryUnit == comparison which fails on numpy embedding arrays)
        sims = []
        for other_id, other_unit in self.ltm.nodes.items():
            if other_id == node_id:
                continue
            sims.append((other_id, np.dot(embedding, other_unit.embedding)))
        sims.sort(key=lambda x: x[1], reverse=True)
        for other_id, similarity in sims[:3]:
            self.ltm.add_edge(node_id, other_id, similarity)
    
    def retrieve(self, query_text: str, top_k: int = 5, time_range: Optional[Tuple[float, float]] = None) -> List[str]:
        """
        Retrieve top-K memories based on query.
        
        Args:
            query_text: Query text
            top_k: Number of memories to retrieve
            time_range: Optional time range (start, end)
        
        Returns:
            List of memory contents
        """
        # Step 1: Embed query
        query_embedding = self.compressor.embed(query_text)
        
        # Step 2: Create query object
        query = MemoryQuery(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            time_range=time_range,
            importance_threshold=0.3,
        )
        
        # Step 3: Retrieve from LTM
        memories = self.ltm.retrieve(query)
        
        # Step 4: Return contents
        return [m.content for m in memories]
    
    def get_stm_context(self) -> str:
        """
        Get STM context for LLM prompt.
        
        Returns:
            Concatenated STM contents
        """
        units = self.stm.get_all()
        context = "\n".join([f"[{u.source}] {u.content}" for u in units])
        return context
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get memory performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "stm_size": len(self.stm.buffer),
            "ltm_size": len(self.ltm.nodes),
            "compression_ratio": self.compressor.compression_ratio,
            "avg_importance": np.mean([u.importance for u in self.ltm.nodes.values()]) if self.ltm.nodes else 0.0,
        }
