from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class SharedState(BaseModel):
    """
    Global state container (Blackboard pattern) for sharing data, 
    context, and references between decoupled agent sub-graphs.
    """
    active_role: Optional[str] = Field(default=None, description="Current active role (e.g., data_scientist, quant_researcher, quant_trader)")
    active_workflow: Optional[str] = Field(default=None, description="Active SOP workflow name")
    
    # Dataset references
    dataset_path: Optional[str] = Field(default=None, description="Path to the active CSV/Parquet dataset")
    target_column: Optional[str] = Field(default=None, description="The target variable name for classification/regression")
    features: List[str] = Field(default_factory=list, description="List of active features in use")
    
    # Model state
    model_type: Optional[str] = Field(default=None, description="Selected model architecture (e.g., RandomForest, Ridge)")
    model_parameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters dictionary")
    model_registry_path: Optional[str] = Field(default=None, description="Registry URI for trained model artifact")
    
    # Validation results
    metrics: Dict[str, float] = Field(default_factory=dict, description="Standard validation metrics (e.g., F1, ROC-AUC, Recall)")
    data_drift_detected: bool = Field(default=False, description="Flag indicating if data drift has been detected")
    target_leakage_detected: bool = Field(default=False, description="Flag indicating if target leakage was flagged during scan")
    
    # Validation history
    validation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Historical decisions and feedback from the Validation Manager")
    
    # Agent execution log
    execution_history: List[str] = Field(default_factory=list, description="Chronological log of tools invoked across sub-graphs")

    # Human-in-the-loop override
    human_override: Optional[str] = Field(default=None, description="Manual override instruction or feedback")
