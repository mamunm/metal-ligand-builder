"""Report generation for metal-ligand binding analysis."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from ..core.data_models import OptimizationResult, Metal, Ligand

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive analysis reports."""
    
    def __init__(self, work_dir: Path):
        """
        Initialize report generator.
        
        Args:
            work_dir: Working directory containing results
        """
        self.work_dir = Path(work_dir)
        self.reports_dir = self.work_dir / "04_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_full_report(
        self,
        metal: Metal,
        ligand: Ligand,
        results: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Path:
        """
        Generate comprehensive HTML report.
        
        Args:
            metal: Metal properties
            ligand: Ligand properties
            results: All analysis results
            config: Configuration used
            
        Returns:
            Path to generated report
        """
        logger.info("Generating comprehensive analysis report...")
        
        # Create report data structure
        report_data = {
            "metadata": self._generate_metadata(metal, ligand, config),
            "summary": self._generate_summary(results),
            "structure_generation": self._analyze_structure_generation(results),
            "optimization_results": self._analyze_optimization(results),
            "orca_preparation": self._analyze_orca_prep(results),
            "energy_analysis": self._analyze_energies(results),
            "best_structures": self._identify_best_structures(results)
        }
        
        # Generate HTML report
        html_content = self._generate_html_report(report_data)
        
        # Save HTML report
        report_path = self.reports_dir / "analysis_report.html"
        report_path.write_text(html_content)
        
        # Also save JSON version
        json_path = self.reports_dir / "analysis_data.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate CSV summaries
        self._generate_csv_summaries(results)
        
        logger.info(f"Report generated: {report_path}")
        
        return report_path
    
    def _generate_metadata(self, metal: Metal, ligand: Ligand, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata section."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "metal": {
                    "symbol": metal.symbol,
                    "charge": metal.charge,
                    "coordination_numbers": metal.coordination_numbers,
                    "preferred_geometries": [g.value for g in metal.preferred_geometries]
                },
                "ligand": {
                    "name": ligand.name,
                    "smiles": ligand.smiles,
                    "charge": ligand.charge,
                    "protonation_state": ligand.protonation_state
                },
                "complex": {
                    "total_charge": metal.charge + ligand.charge,
                    "formula": f"[{metal.symbol}({ligand.name})]^{{{metal.charge + ligand.charge:+d}}}"
                }
            },
            "configuration": {
                "experiment_name": config.get("experiment_name", "Unknown"),
                "max_poses": config.get("max_poses", 0),
                "n_conformers": config.get("n_conformers", 0),
                "xtb_method": config.get("xtb_config", {}).get("method", "Unknown"),
                "xtb_solvent": config.get("xtb_config", {}).get("solvent", "None")
            }
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary."""
        summary = {
            "structures_generated": {
                "ligand_conformers": len(results.get("ligand_conformers", [])),
                "metal_geometries": len(results.get("metal_geometries", [])),
                "complex_poses": len(results.get("complex_poses", [])),
                "binding_sites": len(results.get("binding_sites", []))
            },
            "optimization_summary": {},
            "best_results": {}
        }
        
        # Optimization summary
        if "optimization_results" in results:
            opt_res = results["optimization_results"]
            for struct_type in ["metals", "ligands", "complexes"]:
                if struct_type in opt_res:
                    successful = [r for r in opt_res[struct_type] if r.success]
                    summary["optimization_summary"][struct_type] = {
                        "total": len(opt_res[struct_type]),
                        "successful": len(successful),
                        "failed": len(opt_res[struct_type]) - len(successful)
                    }
        
        # Best structures
        if "optimized_complexes" in results:
            complexes = [r for r in results["optimized_complexes"] if r.success and r.energy]
            if complexes:
                complexes.sort(key=lambda r: r.energy)
                best = complexes[0]
                summary["best_results"]["lowest_energy_complex"] = {
                    "structure": best.optimized_geometry.title,
                    "energy": f"{best.energy:.6f} Hartree",
                    "homo_lumo_gap": f"{best.properties.get('homo_lumo_gap', 0):.3f} eV"
                }
        
        return summary
    
    def _analyze_structure_generation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze structure generation results."""
        analysis = {
            "ligand_conformers": {},
            "binding_sites": {},
            "complex_poses": {}
        }
        
        # Ligand conformer analysis
        if "ligand_conformers" in results:
            conformers = results["ligand_conformers"]
            analysis["ligand_conformers"] = {
                "count": len(conformers),
                "atoms_per_conformer": len(conformers[0].atoms) if conformers else 0
            }
        
        # Binding site analysis
        if "binding_sites" in results:
            sites = results["binding_sites"]
            if isinstance(sites, list) and sites:
                site_types = {}
                for site in sites:
                    if hasattr(site, 'site_type') and hasattr(site.site_type, 'value'):
                        site_type = site.site_type.value
                        site_types[site_type] = site_types.get(site_type, 0) + 1
                
                analysis["binding_sites"] = {
                    "total_sites": len(sites),
                    "site_types": site_types,
                    "average_score": np.mean([s.score for s in sites if hasattr(s, 'score')]) if sites else 0
                }
            else:
                analysis["binding_sites"] = {
                    "total_sites": 0,
                    "site_types": {},
                    "average_score": 0
                }
        
        # Complex pose analysis
        if "complex_poses" in results:
            poses = results["complex_poses"]
            if isinstance(poses, list):
                try:
                    conformers_used = len(set(p.title.split('_conf')[1].split('_')[0] 
                                            for p in poses if hasattr(p, 'title') and '_conf' in p.title))
                except:
                    conformers_used = 0
                
                analysis["complex_poses"] = {
                    "count": len(poses),
                    "conformers_used": conformers_used
                }
            else:
                analysis["complex_poses"] = {
                    "count": 0,
                    "conformers_used": 0
                }
        
        return analysis
    
    def _analyze_optimization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization results."""
        if "optimization_results" not in results:
            return {}
        
        opt_results = results["optimization_results"]
        analysis = {}
        
        for struct_type in ["metals", "ligands", "complexes"]:
            if struct_type not in opt_results:
                continue
            
            type_results = opt_results[struct_type]
            successful = [r for r in type_results if r.success and r.energy]
            
            if not successful:
                continue
            
            energies = [r.energy for r in successful]
            gaps = [r.properties.get("homo_lumo_gap", 0) for r in successful 
                   if r.properties and "homo_lumo_gap" in r.properties]
            
            analysis[struct_type] = {
                "optimization_rate": len(successful) / len(type_results) if type_results else 0,
                "energy_statistics": {
                    "min": min(energies),
                    "max": max(energies),
                    "mean": np.mean(energies),
                    "std": np.std(energies)
                },
                "homo_lumo_gap": {
                    "min": min(gaps) if gaps else 0,
                    "max": max(gaps) if gaps else 0,
                    "mean": np.mean(gaps) if gaps else 0
                }
            }
        
        return analysis
    
    def _analyze_orca_prep(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ORCA preparation results."""
        if "orca_inputs" not in results:
            return {"prepared": False}
        
        orca = results["orca_inputs"]
        
        # Handle case where orca_inputs might not be a dict
        if not isinstance(orca, dict):
            return {"prepared": False}
        
        analysis = {
            "prepared": True,
            "files_generated": {
                "metals": len(orca.get("metal_inputs", [])),
                "ligands": len(orca.get("ligand_inputs", [])),
                "complexes": len(orca.get("complex_inputs", []))
            }
        }
        
        if "summary" in orca:
            analysis["multiplicities"] = orca["summary"].get("multiplicities", [])
            analysis["n_requested"] = orca["summary"].get("n_requested", 0)
        
        return analysis
    
    def _analyze_energies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze energy distributions and trends."""
        analysis = {
            "energy_ranges": {},
            "relative_energies": {}
        }
        
        if "optimization_results" not in results:
            return analysis
        
        opt_results = results["optimization_results"]
        
        # Analyze each structure type
        for struct_type in ["complexes", "ligands", "metals"]:
            if struct_type not in opt_results:
                continue
            
            successful = [r for r in opt_results[struct_type] if r.success and r.energy]
            if not successful:
                continue
            
            # Sort by energy
            successful.sort(key=lambda r: r.energy)
            energies = [r.energy for r in successful]
            
            # Calculate relative energies in kcal/mol
            if energies:
                min_e = min(energies)
                rel_energies = [(e - min_e) * 627.5094740631 for e in energies]
                
                analysis["relative_energies"][struct_type] = {
                    "values": rel_energies[:10],  # Top 10
                    "within_5_kcal": sum(1 for e in rel_energies if e <= 5.0),
                    "within_10_kcal": sum(1 for e in rel_energies if e <= 10.0)
                }
        
        return analysis
    
    def _identify_best_structures(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify best structures from results."""
        best = {}
        
        if "optimization_results" in results:
            opt_results = results["optimization_results"]
            
            # Best complex
            if "complexes" in opt_results:
                complexes = [r for r in opt_results["complexes"] if r.success and r.energy]
                if complexes:
                    complexes.sort(key=lambda r: r.energy)
                    best_complex = complexes[0]
                    best["complex"] = {
                        "title": best_complex.optimized_geometry.title,
                        "energy": best_complex.energy,
                        "properties": best_complex.properties,
                        "file": str(best_complex.output_files.get("optimized_xyz", ""))
                    }
            
            # Best ligand conformer
            if "ligands" in opt_results:
                ligands = [r for r in opt_results["ligands"] if r.success and r.energy]
                if ligands:
                    ligands.sort(key=lambda r: r.energy)
                    best["ligand"] = {
                        "title": ligands[0].optimized_geometry.title,
                        "energy": ligands[0].energy
                    }
        
        return best
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metal-Ligand Binding Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 25px;
            font-weight: bold;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .error {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .formula {{
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Metal-Ligand Binding Analysis Report</h1>
        
        <div class="summary-box">
            <h3>System: <span class="formula">{report_data['metadata']['system']['complex']['formula']}</span></h3>
            <p><strong>Generated:</strong> {report_data['metadata']['timestamp']}</p>
            <p><strong>Experiment:</strong> {report_data['metadata']['configuration']['experiment_name']}</p>
        </div>
        
        <h2>Executive Summary</h2>
        <div>
            <div class="metric">Ligand Conformers: {report_data['summary']['structures_generated']['ligand_conformers']}</div>
            <div class="metric">Complex Poses: {report_data['summary']['structures_generated']['complex_poses']}</div>
            <div class="metric">Binding Sites: {report_data['summary']['structures_generated']['binding_sites']}</div>
        </div>
        
        {self._generate_optimization_section(report_data)}
        
        {self._generate_energy_section(report_data)}
        
        {self._generate_best_structures_section(report_data)}
        
        {self._generate_orca_section(report_data)}
        
        <h2>Configuration Details</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Metal</td><td>{report_data['metadata']['system']['metal']['symbol']} (charge: {report_data['metadata']['system']['metal']['charge']:+d})</td></tr>
            <tr><td>Ligand</td><td>{report_data['metadata']['system']['ligand']['name']} (charge: {report_data['metadata']['system']['ligand']['charge']:+d})</td></tr>
            <tr><td>Complex Charge</td><td>{report_data['metadata']['system']['complex']['total_charge']:+d}</td></tr>
            <tr><td>XTB Method</td><td>{report_data['metadata']['configuration']['xtb_method']}</td></tr>
            <tr><td>Solvent</td><td>{report_data['metadata']['configuration']['xtb_solvent']}</td></tr>
            <tr><td>Max Poses</td><td>{report_data['metadata']['configuration']['max_poses']}</td></tr>
            <tr><td>Conformers</td><td>{report_data['metadata']['configuration']['n_conformers']}</td></tr>
        </table>
    </div>
</body>
</html>"""
        return html
    
    def _generate_optimization_section(self, report_data: Dict[str, Any]) -> str:
        """Generate optimization results section."""
        if "optimization_results" not in report_data:
            return ""
        
        opt_data = report_data["optimization_results"]
        
        html = "<h2>Optimization Results</h2><table><tr><th>Structure Type</th><th>Total</th><th>Successful</th><th>Success Rate</th><th>Min Energy (Hartree)</th></tr>"
        
        for struct_type in ["metals", "ligands", "complexes"]:
            if struct_type in opt_data:
                data = opt_data[struct_type]
                total = int(1 / data["optimization_rate"]) if data["optimization_rate"] > 0 else 0
                successful = int(total * data["optimization_rate"])
                rate = f"{data['optimization_rate']*100:.1f}%"
                min_e = f"{data['energy_statistics']['min']:.6f}"
                
                html += f"<tr><td>{struct_type.capitalize()}</td><td>{total}</td><td class='success'>{successful}</td><td>{rate}</td><td>{min_e}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_energy_section(self, report_data: Dict[str, Any]) -> str:
        """Generate energy analysis section."""
        if "energy_analysis" not in report_data:
            return ""
        
        energy_data = report_data["energy_analysis"]
        
        html = "<h2>Energy Analysis</h2>"
        
        if "relative_energies" in energy_data:
            html += "<h3>Relative Energies (kcal/mol)</h3><table><tr><th>Structure Type</th><th>Within 5 kcal/mol</th><th>Within 10 kcal/mol</th></tr>"
            
            for struct_type, data in energy_data["relative_energies"].items():
                html += f"<tr><td>{struct_type.capitalize()}</td><td>{data['within_5_kcal']}</td><td>{data['within_10_kcal']}</td></tr>"
            
            html += "</table>"
        
        return html
    
    def _generate_best_structures_section(self, report_data: Dict[str, Any]) -> str:
        """Generate best structures section."""
        if "best_structures" not in report_data or not report_data["best_structures"]:
            return ""
        
        best = report_data["best_structures"]
        
        html = "<h2>Best Structures</h2>"
        
        if "complex" in best:
            complex_data = best["complex"]
            html += f"""<div class="summary-box">
                <h3>Lowest Energy Complex</h3>
                <p><strong>Structure:</strong> {complex_data['title']}</p>
                <p><strong>Energy:</strong> {complex_data['energy']:.6f} Hartree</p>
                <p><strong>HOMO-LUMO Gap:</strong> {complex_data['properties'].get('homo_lumo_gap', 0):.3f} eV</p>
            </div>"""
        
        return html
    
    def _generate_orca_section(self, report_data: Dict[str, Any]) -> str:
        """Generate ORCA preparation section."""
        if "orca_preparation" not in report_data or not report_data["orca_preparation"].get("prepared"):
            return ""
        
        orca_data = report_data["orca_preparation"]
        
        html = "<h2>ORCA Input Preparation</h2><div class='summary-box'>"
        html += f"<p><strong>Files Generated:</strong></p><ul>"
        
        for struct_type, count in orca_data["files_generated"].items():
            if count > 0:
                html += f"<li>{struct_type.capitalize()}: {count} files</li>"
        
        if "multiplicities" in orca_data:
            html += f"<li>Multiplicities considered: {', '.join(map(str, orca_data['multiplicities']))}</li>"
        
        html += "</ul></div>"
        
        return html
    
    def _generate_csv_summaries(self, results: Dict[str, Any]) -> None:
        """Generate CSV summary files."""
        # Optimization summary
        if "optimization_results" in results:
            for struct_type in ["metals", "ligands", "complexes"]:
                if struct_type not in results["optimization_results"]:
                    continue
                
                data = []
                for i, result in enumerate(results["optimization_results"][struct_type]):
                    if result.success and result.optimized_geometry:
                        row = {
                            "index": i,
                            "structure": result.optimized_geometry.title,
                            "energy": result.energy,
                            "success": result.success,
                            "homo_lumo_gap": result.properties.get("homo_lumo_gap", None),
                            "dipole_moment": result.properties.get("dipole_moment", None)
                        }
                        data.append(row)
                
                if data:
                    df = pd.DataFrame(data)
                    df.sort_values("energy", inplace=True)
                    csv_path = self.reports_dir / f"{struct_type}_summary.csv"
                    df.to_csv(csv_path, index=False)
    
    def save_results_archive(self, results: Dict[str, Any]) -> Path:
        """
        Save complete results archive.
        
        Args:
            results: All analysis results
            
        Returns:
            Path to results archive
        """
        archive_path = self.work_dir / "06_metadata_files" / "results_archive.json"
        
        # Create serializable version of results
        serializable_results = self._make_serializable(results)
        
        with open(archive_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results archive saved: {archive_path}")
        
        return archive_path
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # Convert objects to dict representation
            return {
                '_type': obj.__class__.__name__,
                'data': self._make_serializable(obj.__dict__)
            }
        else:
            return obj