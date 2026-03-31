




__all__ = ["equivariant-transformer", "egnn", "painn", "equivariant-transformerf2d"]
self.init_e = EdgeFeatureInit(self.distance_expansion, act_class, num_rbf, hidden_channels)

self.motif_agg = BRICSMotifAggregation(self.hidden_channels)
self.influence_matrix = InfluenceMatrix()
self.long_range_motif = LongRangeMotifInteraction(self.hidden_channels)


self.energy_sentinel = EnergySentinel(lambda_penalty=1.0)
self.potential_energy = MolecularPotential(cutoff=self.cutoff_upper)