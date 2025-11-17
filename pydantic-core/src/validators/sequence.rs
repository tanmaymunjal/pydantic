use std::sync::Arc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBytes, PyDict, PyList, PyString, PyTuple};

use crate::errors::{ErrorType, ValError, ValResult};
use crate::input::Input;

use super::{build_validator, BuildValidator, CombinedValidator, DefinitionsBuilder, ValidationState, Validator};

#[derive(Debug)]
pub struct SequenceValidator {
    item_validator: Option<Arc<CombinedValidator>>,
    name: String,
}

impl BuildValidator for SequenceValidator {
    const EXPECTED_TYPE: &'static str = "sequence";

    fn build(
        schema: &Bound<'_, PyDict>,
        config: Option<&Bound<'_, PyDict>>,
        definitions: &mut DefinitionsBuilder<Arc<CombinedValidator>>,
    ) -> PyResult<Arc<CombinedValidator>> {
        let py = schema.py();
        let item_validator = match schema.get_item(intern!(py, "items_schema"))? {
            Some(d) => {
                let validator = build_validator(&d, config, definitions)?;
                match validator.as_ref() {
                    CombinedValidator::Any(_) => None,
                    _ => Some(validator),
                }
            }
            None => None,
        };

        let name = match &item_validator {
            Some(v) => format!("sequence[{}]", v.get_name()),
            None => "sequence[any]".to_string(),
        };

        Ok(CombinedValidator::Sequence(Self { item_validator, name }).into())
    }
}

impl_py_gc_traverse!(SequenceValidator { item_validator });

impl SequenceValidator {
    /// Extract a sequence iterable from a Python object, similar to extract_sequence_iterable
    /// but handles the sequence-specific logic (rejecting str/bytes)
    fn extract_sequence<'py>(obj: &Bound<'py, PyAny>) -> ValResult<(Bound<'py, PyList>, SequenceType)> {
        let py = obj.py();

        // Reject str/bytes - they're sequences but shouldn't be validated as such
        if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyBytes>() {
            let type_name = obj.get_type().qualname()?.to_string();
            let context = PyDict::new(py);
            context.set_item("type_name", type_name.as_str())?;
            return Err(ValError::new(
                ErrorType::CustomError {
                    error_type: "sequence_str".to_string(),
                    message_template: format!("'{type_name}' instances are not allowed as a Sequence value"),
                    context: Some(context.into()),
                },
                obj,
            ));
        }

        // Handle concrete types
        if let Ok(list) = obj.downcast::<PyList>() {
            return Ok((list.clone(), SequenceType::List));
        } else if let Ok(tuple) = obj.downcast::<PyTuple>() {
            // Convert tuple to list for validation
            let items: Vec<Py<PyAny>> = tuple.iter().map(Bound::unbind).collect();
            let py_list = PyList::new(py, items)?;
            return Ok((py_list, SequenceType::Tuple));
        }

        // Check for deque
        let collections_module = py.import("collections")?;
        let deque_type = collections_module.getattr("deque")?;
        if obj.is_instance(&deque_type)? {
            // Extract items and maxlen from deque
            // maxlen is always an attribute on deque, but can be None if no maxlen was specified
            let maxlen = match obj.getattr("maxlen") {
                Ok(attr) => {
                    if PyAnyMethods::is_none(&attr) {
                        None
                    } else {
                        attr.extract::<isize>().ok()
                    }
                }
                _ => None,
            };
            let mut items = Vec::new();
            if let Ok(iter) = obj.try_iter() {
                for item_result in iter {
                    let item = item_result.map_err(|e| {
                        ValError::new(
                            ErrorType::IterationError {
                                error: format!("{e}"),
                                context: None,
                            },
                            obj,
                        )
                    })?;
                    items.push(item.unbind());
                }
            }
            let py_list = PyList::new(py, items)?;
            return Ok((py_list, SequenceType::Deque { maxlen }));
        }

        // For other sequence types, try to convert to list
        if let Ok(iter) = obj.try_iter() {
            let mut items = Vec::new();
            for item_result in iter {
                let item = item_result.map_err(|e| {
                    ValError::new(
                        ErrorType::IterationError {
                            error: format!("{e}"),
                            context: None,
                        },
                        obj,
                    )
                })?;
                items.push(item.unbind());
            }
            let py_list = PyList::new(py, items)?;
            // We can't determine the original type for arbitrary iterables, so return as List
            return Ok((py_list, SequenceType::List));
        }

        Err(ValError::new(ErrorType::IterableType { context: None }, obj))
    }

    /// Reconstruct the original sequence type from validated items
    fn reconstruct_sequence(
        py: Python<'_>,
        seq_type: SequenceType,
        validated_items: Vec<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        match seq_type {
            SequenceType::List => Ok(PyList::new(py, validated_items)?.into()),
            SequenceType::Tuple => Ok(PyTuple::new(py, validated_items)?.into()),
            SequenceType::Deque { maxlen } => {
                let collections_module = py.import("collections")?;
                let deque_type = collections_module.getattr("deque")?;
                let items_list = PyList::new(py, validated_items)?;
                let deque = if let Some(maxlen_val) = maxlen {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("maxlen", maxlen_val)?;
                    deque_type.call((items_list,), Some(&kwargs))?
                } else {
                    deque_type.call1((items_list,))?
                };
                Ok(deque.into())
            }
        }
    }
}

#[derive(Debug, Clone)]
enum SequenceType {
    List,
    Tuple,
    Deque { maxlen: Option<isize> },
}

impl Validator for SequenceValidator {
    fn validate<'py>(
        &self,
        py: Python<'py>,
        input: &(impl Input<'py> + ?Sized),
        state: &mut ValidationState<'_, 'py>,
    ) -> ValResult<Py<PyAny>> {
        let Some(obj) = input.as_python() else {
            return Err(ValError::new(
                ErrorType::NeedsPythonObject {
                    context: None,
                    method_name: "sequence".to_string(),
                },
                input,
            ));
        };

        // Extract sequence and determine original type
        let (py_list, seq_type) = Self::extract_sequence(obj)?;
        let actual_length = py_list.len();

        // Validate items
        let validated_items = match &self.item_validator {
            Some(ref item_validator) => {
                let mut validated = Vec::with_capacity(actual_length);
                for (index, item) in py_list.iter().enumerate() {
                    match item_validator.validate(py, &item, state) {
                        Ok(validated_item) => validated.push(validated_item),
                        Err(ValError::LineErrors(line_errors)) => {
                            // Convert line errors to have correct index
                            let mut errors = Vec::new();
                            for err in line_errors {
                                errors.push(err.with_outer_location(index));
                            }
                            return Err(ValError::LineErrors(errors));
                        }
                        Err(err) => return Err(err),
                    }
                }
                validated
            }
            None => {
                // No item validator, just return the items as-is
                py_list.iter().map(Bound::unbind).collect()
            }
        };

        // Reconstruct the original sequence type
        Self::reconstruct_sequence(py, seq_type, validated_items).map_err(|e| {
            ValError::new(
                ErrorType::CustomError {
                    error_type: "sequence_reconstruction".to_string(),
                    message_template: e.to_string(),
                    context: None,
                },
                input,
            )
        })
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}
