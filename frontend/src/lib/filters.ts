export interface FilterState {
  search: string
  store_id: string
  cat_id: string
  demand_class: string
}

export const EMPTY_FILTERS: FilterState = { search: '', store_id: '', cat_id: '', demand_class: '' }
