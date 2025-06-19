import os
import json
import asyncio
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import httpx
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

@dataclass
class ProjectInfo:
    name: str
    url: str
    stars: int
    language: str
    size_mb: float
    description: str

@dataclass
class TopicAnalysis:
    topic: str
    projects: Dict[str, Dict[str, List[ProjectInfo]]]

class PopularTopicsAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.github_token = os.getenv('GITHUB_TOKEN') 
        print(f"GITHUB_TOKEN: {self.github_token}")
        self.sizes = ['small', 'medium', 'large']
        self.languages = ['Python', 'Java', 'Rust', 'C++', 'C', 'Rust', 'Javascript', 'PHP']
        
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        self.data_dir = project_root / '.data'
        self.data_dir.mkdir(exist_ok=True)
        
    async def get_popular_topics(self, n: int) -> List[str]:
        """Use LLM to get n most popular programming topics."""
        
        try:
            # Define the function for LLM to call
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_popular_topics",
                        "description": "Get the most popular programming and technology topics",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "topics": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": f"List of {n} most popular programming/technology topics"
                                }
                            },
                            "required": ["topics"]
                        }
                    }
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini", 
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in programming and technology trends. You must provide exactly the requested number of popular programming topics."
                    },
                    {
                        "role": "user",
                        "content": f"Please provide exactly {n} of the most popular programming and software development topics right now. (e.g. Machine Learning, Cybersecurity, Blockchain, etc.)"
                    }
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "get_popular_topics"}}
            )
            
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and tool_calls[0].function.name == "get_popular_topics":
                result = json.loads(tool_calls[0].function.arguments)
                topics = result.get("topics", [])
                
                valid_topics = [topic.strip() for topic in topics if topic and topic.strip()]
                print(f"Valid topics: {valid_topics}")
                if len(valid_topics) >= n:
                    final_topics = valid_topics[:n]
                    print(f"Found {len(final_topics)} popular topics: {final_topics}")
                    return final_topics
                else:
                    print(f"LLM returned only {len(valid_topics)} topics")
                    return valid_topics
            else:
                print("Function call failed")
                raise Exception("Function call failed")
                
        except Exception as e:
            print(f"Error getting topics from LLM: {e}")
            print("Using default popular topics")
            return []

    def get_size_query(self, size: str) -> str:
        """Convert size to GitHub search query parameters."""
        if size == 'small':
            return 'size:<1000'
        elif size == 'medium':
            return 'size:1000..10000'
        else:
            return 'size:>10000'

    async def search_github_projects(self, topic: str, size: str, language: str, per_page: int = 2) -> List[ProjectInfo]:
        """Search GitHub for projects matching topic, size, and language."""
        
        size_query = self.get_size_query(size)
        query = f"{topic} language:{language} {size_query} stars:>10"
        
        url = "https://api.github.com/search/repositories"
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': per_page
        }
        
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'PopularTopicsAnalyzer/1.0'
        }
        
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                projects = []
                
                for repo in data.get('items', [])[:2]:
                    project = ProjectInfo(
                        name=repo['full_name'],
                        url=repo['html_url'],
                        stars=repo['stargazers_count'],
                        language=repo.get('language', 'Unknown'),
                        size_mb=repo.get('size', 0) / 1024,
                        description=repo.get('description', 'No description')
                    )
                    projects.append(project)
                
                return projects
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                print(f"Rate limit exceeded for {topic}/{size}/{language}")
                return []
            else:
                print(f"HTTP error {e.response.status_code} for {topic}/{size}/{language}: {e}")
                return []
        except Exception as e:
            print(f"Error searching GitHub for {topic}/{size}/{language}: {e}")
            return []

    async def analyze_topic(self, topic: str) -> TopicAnalysis:
        """Analyze a single topic across all sizes and languages."""
        print(f"\nAnalyzing topic: {topic}")
        
        analysis = TopicAnalysis(topic=topic, projects={})
        
        for size in self.sizes:
            analysis.projects[size] = {}
            for language in self.languages:
                analysis.projects[size][language] = []
        
        
        if not hasattr(self, 'global_request_count'):
            self.global_request_count = 0
        if not hasattr(self, 'batch_size'):
            self.batch_size = 30 
        if not hasattr(self, 'wait_time'):
            self.wait_time = 60
        
        total_requests = len(self.sizes) * len(self.languages)
        completed = 0
        
        for size in self.sizes:
            for language in self.languages:
                try:
                    result = await self.search_github_projects(topic, size, language)
                    analysis.projects[size][language] = result
                    completed += 1
                    self.global_request_count += 1
                    
                    if result:
                        print(f"  {size} {language}: {len(result)} projects found")
                    else:
                        print(f"  {size} {language}: no projects found")
                    
                    if self.global_request_count >= self.batch_size and completed < total_requests:
                        print(f"  Completed {self.global_request_count} requests, waiting {self.wait_time}s for rate limit reset...")
                        await asyncio.sleep(self.wait_time)
                        self.global_request_count = 0
                        print(f"  Resuming requests...")
                    else:
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    print(f"  Error for {topic}/{size}/{language}: {e}")
                    analysis.projects[size][language] = []
        
        total_found = sum(len(projects) for size_dict in analysis.projects.values() 
                         for projects in size_dict.values())
        print(f"Topic '{topic}' complete: {total_found} total projects found")
        
        return analysis

    def create_spec_file(self, analyses: List[TopicAnalysis], output_path: str = "topics_repos.spec"):
        """Create a spec file for bench_repos.py from the analysis results."""
        output_path = self.data_dir / output_path
        
        spec_lines = []
        
        for analysis in analyses:
            spec_lines.append(f"# Topic: {analysis.topic}")
            
            for size in self.sizes:
                for language in self.languages:
                    projects = analysis.projects[size][language]
                    for project in projects:
                        git_url = project.url.replace('https://github.com/', 'https://github.com/') + '.git'
                        spec_lines.append(git_url)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(spec_lines))
        
        print(f"Created spec file: {output_path}")
        return output_path

    def print_analysis_results(self, analyses: List[TopicAnalysis]):
        """Print formatted analysis results."""
        print("\n" + "="*80)
        print("POPULAR TOPICS ANALYSIS RESULTS")
        print("="*80)
        
        for analysis in analyses:
            print(f"\nTOPIC: {analysis.topic.upper()}")
            print("-" * 60)
            
            for size in self.sizes:
                print(f"\n  {size.upper()} Projects:")
                
                for language in self.languages:
                    projects = analysis.projects[size][language]
                    if projects:
                        print(f"\n    {language}:")
                        for i, project in enumerate(projects, 1):
                            print(f"      {i}. {project.name}")
                            print(f"         {project.stars:,} stars")
                            print(f"         {project.size_mb:.1f} MB")
                            print(f"         {project.url}")
                            if project.description:
                                print(f"         {project.description[:80]}...")

    def save_results_json(self, analyses: List[TopicAnalysis], output_path: str = "topics_analysis.json"):
        """Save analysis results to JSON file in flat format: Topic + size + language + github repo name."""
        output_path = self.data_dir / output_path
        
        results = []
        
        for analysis in analyses:
            for size in self.sizes:
                for language in self.languages:
                    projects = analysis.projects[size][language]
                    for project in projects:
                        result_entry = {
                            'topic': analysis.topic,
                            'size': size,
                            'language': language,
                            'github_repo_name': project.name,
                            'url': project.url,
                            'stars': project.stars,
                            'size_mb': project.size_mb,
                            'description': project.description
                        }
                        results.append(result_entry)
        
        results.sort(key=lambda x: (x['topic'], x['size'], x['language'], -x['stars']))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
        print(f"Total entries: {len(results)}")

    def print_simple_results(self, analyses: List[TopicAnalysis]):
        """Print results in simple Topic + size + language + github repo name format."""
        print("\n" + "="*80)
        print("RESULTS: Topic + Size + Language + GitHub Repo Name")
        print("="*80)
        
        for analysis in analyses:
            for size in self.sizes:
                for language in self.languages:
                    projects = analysis.projects[size][language]
                    for project in projects:
                        print(f"{analysis.topic} | {size} | {language} | {project.name} | {project.stars:,} stars")

    async def run_analysis(self, n: int):
        """Run the complete analysis workflow."""
        print(f"Starting Popular Topics Analysis for {n} topics...")
        
        topics = await self.get_popular_topics(n)
        
        self.global_request_count = 0
        self.batch_size = 30 
        self.wait_time = 60
        
        analyses = []
        for i, topic in enumerate(topics):
            analysis = await self.analyze_topic(topic)
            analyses.append(analysis)
            
            if self.global_request_count >= self.batch_size and i < len(topics) - 1:
                print(f"\nCompleted {self.global_request_count} total requests, waiting {self.wait_time}s before next topic...")
                await asyncio.sleep(self.wait_time)
                self.global_request_count = 0
                print(f"Resuming with next topic...")
        
        self.print_simple_results(analyses)
        self.save_results_json(analyses)
        
        spec_file = self.create_spec_file(analyses)
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {self.data_dir / 'topics_analysis.json'}")
        print(f"Spec file created: {spec_file}")
        print(f"If needed, you can use: python -m codelib.utils.bench_repos {spec_file}")
        
        return analyses

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='Number of popular topics to analyze')
    
    args = parser.parse_args()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key not provided.")
        return
    
    analyzer = PopularTopicsAnalyzer(api_key)
    await analyzer.run_analysis(args.n)

if __name__ == "__main__":
    asyncio.run(main()) 