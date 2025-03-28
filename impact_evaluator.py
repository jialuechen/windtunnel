class ImpactEvaluator:
    def evaluate(self, strategy_result):
        executions = strategy_result["executions"]
        total_size = sum([e["size"] for e in executions])
        avg_price = sum([e["executed_price"] * e["size"] for e in executions]) / total_size
        return {
            "total_volume": total_size,
            "avg_execution_price": avg_price
        }
